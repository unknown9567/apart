'''

Reference:
Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Forked from
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

'''


import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = \
            (in_planes != out_planes or stride != 1) and \
            nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

    def residual(self, x):
        x = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return x, out

    def forward(self, x):
        if self.shortcut:
            x, out = self.residual(x)
            x = self.shortcut(x)
        else:
            _, out = self.residual(x)
        return x + out


class BlockList(nn.Sequential):

    def __init__(self, num_layers, in_planes, out_planes,
                 block, stride, drop_rate=0.0):
        super(BlockList, self).__init__(*[block(
            i == 0 and in_planes or out_planes, out_planes,
            i == 0 and stride or 1, drop_rate
        ) for i in range(num_layers)])


class WideResNet(nn.Module):

    def __init__(self, depth, num_classes, widen_factor=1,
                 drop_rate=0.0, block=BasicBlock):
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        super(WideResNet, self).__init__()
        planes = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, planes[0], 3, 1, 1, bias=False)
        self.block1 = BlockList(n, planes[0], planes[1], block, 1, drop_rate)
        self.block2 = BlockList(n, planes[1], planes[2], block, 2, drop_rate)
        self.block3 = BlockList(n, planes[2], planes[3], block, 2, drop_rate)
        self.bn = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(x.size(0), -1)
        return self.fc(out)


def WideResNet16_2(num_classes=10, block=BasicBlock):
    return WideResNet(16, num_classes, 2, block=block)


def WideResNet40_1(num_classes=10, block=BasicBlock):
    return WideResNet(40, num_classes, 1, block=block)


def WideResNet40_2(num_classes=10, block=BasicBlock):
    return WideResNet(40, num_classes, 2, block=block)


def WideResNet28_10(num_classes=10, block=BasicBlock):
    return WideResNet(28, num_classes, 10, block=block)


__all__ = list(name for name in globals() if 'wideresnet' in name.lower())