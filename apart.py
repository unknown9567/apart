from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['APARTv2']


# Rescale the unbiased estimator to get a biased one
def rescale_batch_norm_var(x, var, inplace=False):
    size = float(x[:, 0, ...].nelement())
    ratio = (size - 1) / size
    if inplace:
        return var.data.copy_(ratio * var)
    return ratio * var


# There is potentially a bug in Pytorch's implementation for BN,
# which seems to update the statistics by unbiased estimators
# but normalize BN's inputs by the biased estimators during training
# Check if there exists this bug
def check_batch_norm_unbiased_parameter():
    size = 8
    x = torch.arange(size).reshape(2, 1, int((size // 2) ** 0.5), -1).float()
    mean, var = torch.zeros(1), torch.zeros(1)
    y1 = F.batch_norm(x, mean, var, None, None, True, 1.0, 1e-5)
    y2 = F.batch_norm(x, mean, var, None, None, False, 1.0, 1e-5)
    y3 = F.batch_norm(x, mean, rescale_batch_norm_var(x, var),
                      None, None, False, 1.0, 1e-5)
    if (y1 - y2).abs().mean().item() < 1e-5:
        return False
    elif (y1 - y3).abs().mean().item() < 1e-5:
        return True
    else:
        raise Exception('Unknown bug of batch norm`s unbiased parameter')
RESCALE_VAR = check_batch_norm_unbiased_parameter()


class Hook(nn.Module):

    def __init__(self, batch_norm, epsilon=0.1, groups=0):
        super(Hook, self).__init__()
        self.epsilon, self.groups = epsilon, groups
        self.register_buffer('cache_running_mean',
                             torch.zeros_like(batch_norm.running_mean))
        self.register_buffer('cache_running_var',
                             torch.zeros_like(batch_norm.running_var))
        self.delta_mean = \
            nn.Parameter(torch.zeros_like(batch_norm.running_mean))
        self.delta_var = \
            nn.Parameter(torch.zeros_like(batch_norm.running_var))
        self.mode = 'original'
        self.affine_parameters = []
        self.track_running_stats = None
        self.register_batch_norm(batch_norm)

    def register_batch_norm(self, batch_norm):
        batch_norm.register_forward_hook(self.forward_hook)
        batch_norm.register_forward_pre_hook(self.forward_pre_hook)
        return batch_norm

    def forward_pre_hook(self, module, input):
        if self.mode == 'original':
            return
        assert module.training, 'Not training'
        assert self.mode in ['proxy', 'adver'], f'Unknown mode: {self.mode}'

        if module.affine:
            self.affine_parameters.append((module.weight, module.bias))
            delattr(module, 'weight')
            delattr(module, 'bias')
            module.register_parameter('weight', None)
            module.register_parameter('bias', None)

        self.cache_running_mean.data.copy_(module.running_mean)
        self.cache_running_var.data.copy_(module.running_var)
        self.track_running_stats = module.track_running_stats
        module.track_running_stats = True
        if self.mode == 'adver' and self.groups:
            input = input[0]
            size = input.size(0)
            groups = self.groups if self.groups > 0 else size
            assert size % groups == 0, f'The batch with {size} samples ' \
                                       f'cannot be split into {groups} groups'
            input = input.reshape(size // groups, groups, *input.size()[1:])
            input = input.reshape(size // groups, -1, *input.size()[3:])
            running_mean, running_var = module.running_mean, module.running_var
            running_mean.data.zero_(), running_var.data.zero_()
            delattr(module, 'running_mean')
            delattr(module, 'running_var')
            module.register_buffer('running_mean', running_mean.view(1, -1).
                                   repeat(groups, 1).view(-1))
            module.register_buffer('running_var', running_var.view(1, -1).
                                   repeat(groups, 1).view(-1))
            return input

    def forward_hook(self, module, input, output):
        if self.mode == 'original':
            return

        delta_mean, delta_var = self.delta_mean, self.delta_var
        if self.mode == 'proxy':
            delta_mean.data.zero_(), delta_var.data.zero_()
        elif self.mode == 'adver':
            if delta_mean.grad is not None:
                delta_mean.data.copy_(
                    delta_mean.grad.sign().clip(-self.epsilon, self.epsilon)
                )
                delta_mean.grad = None
            if delta_var.grad is not None:
                delta_var.data.copy_(
                    delta_var.grad.sign().clip(-self.epsilon, self.epsilon)
                )
                delta_var.grad = None

        exponential_average_factor = self.get_exponential_average_factor(module)
        if self.mode == 'proxy' or (self.mode == 'adver' and self.groups == 0):
            mean = (module.running_mean - (1 - exponential_average_factor) *
                    self.cache_running_mean) / exponential_average_factor
            var = (module.running_var - (1 - exponential_average_factor) *
                   self.cache_running_var) / exponential_average_factor
        if self.mode == 'adver':
            if self.groups:
                groups = self.groups if self.groups > 0 else \
                    (output.size(1) // module.num_features)
                mean = module.running_mean / exponential_average_factor
                var = module.running_var / exponential_average_factor
                delattr(module, 'running_mean')
                delattr(module, 'running_var')
                module.register_buffer('running_mean',
                                       self.cache_running_mean.detach().clone())
                module.register_buffer('running_var',
                                       self.cache_running_var.detach().clone())
                delta_mean = delta_mean.view(1, -1).repeat(groups, 1).view(-1)
                delta_var = delta_var.view(1, -1).repeat(groups, 1).view(-1)
            else:
                module.running_mean.data.copy_(self.cache_running_mean)
                module.running_var.data.copy_(self.cache_running_var)

        if RESCALE_VAR:
            var = rescale_batch_norm_var(output, var)
        module.track_running_stats = self.track_running_stats

        broadcasting = (1, -1) + (1, ) * (output.dim() - 2)
        shift = (delta_mean * mean / (var + module.eps).sqrt()).view(broadcasting)
        scale = (1 + delta_var).view(broadcasting)
        output = scale * (output - shift)

        if self.mode == 'adver' and self.groups:
            output = output.reshape(output.size(0), groups, -1, *output.size()[2:])
            output = output.reshape(-1, *output.size()[2:])

        if module.affine:
            weight, bias = self.affine_parameters.pop()
            delattr(module, 'weight')
            delattr(module, 'bias')
            module.weight = weight
            module.bias = bias
            output = weight.view(broadcasting) * output + bias.view(broadcasting)
        return output

    # Follow Pytorch's implementation to get the exponential average factor
    @staticmethod
    def get_exponential_average_factor(module):
        if module.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = module.momentum
        if module.training and module.track_running_stats:
            if module.num_batches_tracked is not None:
                if module.momentum is None:
                    exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                else:
                    exponential_average_factor = module.momentum
        return exponential_average_factor


class APART(nn.ModuleList):

    def __init__(self, model, epsilon=0.1, groups=0):
        super(APART, self).__init__(list(map(
            lambda bn: Hook(bn, epsilon, groups),
            filter(lambda l: isinstance(l, _BatchNorm), model.modules())
        )))

    def set_mode(self, mode):
        for hook in self:
            hook.mode = mode
        return self

    def set_epsilon(self, epsilon):
        for hook in self:
            hook.epsilon = epsilon
        return self

    @contextmanager
    def to_proxy(self):
        yield self.set_mode('proxy')
        self.set_mode('original')

    @contextmanager
    def to_adver(self):
        yield self.set_mode('adver')
        self.set_mode('original')

    def register_batch_norm(self, model):
        batch_norms = list(filter(
            lambda l: isinstance(l, _BatchNorm), model.modules()))
        assert len(batch_norms) == len(self)
        for batch_norm, apart_hook in zip(batch_norms, self):
            apart_hook.register_batch_norm(batch_norm)
