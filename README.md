# Adversarially Perturbed Batch Normalization: A Simple Way to Improve Image Recognition

This repository contains the PyTorch code for the paper
> Adversarially Perturbed Batch Normalization: A Simple Way to Improve Image Recognition

**A**dversarially **P**erturbed b**A**tch no**R**maliza**T**ion (**APART**) enhances models' robustness against 
noise in statistics of Batch Normalization (BN) to improve their generalization. 

Implementation-wise, APART class in the file `apart.py` allows users to deploy APART by a few lines. 
The file `simple_example.py` provides an example of deploying `APART` with additional **FOUR** lines
to improve a CIFAR model.
Also, this repository contains the code in `main` for reproducing the results in the paper.


## Setup

**Environment**:

- [PyTorch 1.10.2](https://pytorch.org/)
- [PyTorch-Lightning 1.5.10](https://www.pytorchlightning.ai/) (Optional, required by reproducing the complete experiments)
- [PyYAML 6.0](https://pyyaml.org/) (Optional, required by reproducing the complete experiments)


## Usage

APART computes two forward-backward passes at each step of gradient descent methods. 
The weighted sum of the gradients in these two steps is eventually the gradient for 
updating the parameters. Meanwhile, Pytorch allows gradient accumulation just by 
keeping `grad` in the variables without employing the method `zero_grad` of `torch.nn.Module`. 
In summary, just keep in mind that APART requires two backward passes over the same samples without `zero_grad`
the parameters' gradients:
```python
from apart import APART
# get the dataloader for your dataset
# the last incomplete batch in each epoch should 
# be dropped (set drop_last=True in PyTorch's DataLoader) 
# if APART's argument `groups` > 1 
dataloader = ... 
model = YourModel() # should include Batch Normalization layers
optimizer = ... # instantiate a regular optimizer for your model
apart = APART(model, epsilon, groups) # instantiate APART
N = ... # set number of samples used in APART's second step, <= batch_size
r = batch_size / (batch_size + N) # the weight of the first loss
...

for x, y in dataloader:
    
    optimizer.zero_grad() # clear gradients stored in each parameter
    
    # first step of APART
    with apart.to_proxy(): # prepare for the first step
        loss = loss_function(model(x), y) # normal forward pass
        (r * loss).backward() # normal backward pass
        
    # second step of APART
    with apart.to_adver(): # prepare for the second step
        # adversarial forward and backward passes over a subset of this batch
        ((1 - r) * loss_function(model(x[:N]), y[:N])).backward()
        
    optimizer.zero_grad() # perform the regular gradient update
    
    ... # log the loss
    
```


## Documentation

#### `APART.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `model` (torch.nn.Module) | PyTorch model with Batch Normalization layers |
| `epsilon` (float) | perturbation radius of APART (default: 0.1) |
| `groups` (int) | group number of APART; `batch_size` in dataloader should be divisible by `groups`; `groups=1` is equivalent to `groups=0` *(default: 0)* |


#### `APART.to_proxy`

No arguments, employed in APART's first step:
```python
with apart.to_proxy():
    # `apart` will implicitly store the gradients required by APART's attacks
    ... # perform forward and backward passes
```


#### `APART.to_adver`

No arguments, employed in APART's second step:
```python
with apart.to_adver():
    # `apart` will implicitly perturb BN statistics
    ... # perform forward and backward passes
```


<br>