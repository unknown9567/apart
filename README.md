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
- [PyTorch-Lightning 1.5.10](https://www.pytorchlightning.ai/) (Optional, required by reproducing the complete experiments in `main`)
- [PyYAML 6.0](https://pyyaml.org/) (Optional, required by reproducing the complete experiments in `main`)


## Usage

APART computes two forward-backward passes at each step of gradient descent methods. 
The weighted sum of the gradients in these two steps is eventually the gradient for 
updating the parameters. Meanwhile, Pytorch uses gradient accumulation allowed by PyTorch. 
In summary, just keep in mind that APART requires two backward passes over the same samples without 
deploying `zero_grad` in the intermediate step between these two passes:
```python
from apart import APART
# get the dataloader for your dataset
# if APART's argument `groups` > 1 
# the last incomplete batch in each epoch should 
# be dropped (set drop_last=True in PyTorch's DataLoader) 
dataloader = ... 
model = YourModel() # should include Batch Normalization layers
optimizer = ... # instantiate a regular optimizer for your model
apart = APART(model, epsilon, groups) # instantiate APART
N = ... # set number of samples used in APART's second step, <= batch_size
r = batch_size / (batch_size + N) # the weight of the loss in the first step
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


## How to train

Once the repo is cloned, experiments can be run using 
`simple_example.py` or `main/train_*_*.py`.

### Simple Example

`simple_example.py` shows how to deploy `APART` class in `apart.py` by a simple example, 
which also contains the code implementing a standard training pipeline and provides 
the comparison between the standard method and APART.

run `simple_example.py`:
```python
python3 --dataset cifar100 --model resnet18 --epsilon 0.1 --groups 16
```

### Main Experiments

`main` contains the code for reproducing main experiments in the paper, 
including the experiments of the standard method, APART and APART-SAM over
CIFAR-10, CIFAR-100, Tiny-ImageNet and ImageNet.

#### CIFAR-10 & CIFAR-100

run `main/train_cifar_standard.py` for the standard method:
```python
python main/train_cifar_standard.py --dataset cifar10/cifar100 --model wideresnet40_2/preact_resnet
```

run `main/train_cifar_apart.py` for APART:
```python
python main/train_cifar_apart.py --dataset cifar10/cifar100 --model wideresnet40_2/preact_resnet \
--ratio 1 --epsilon 0.1 --groups 16
```

run `main/train_cifar_apart_sam.py` for APART-SAM:
```python
python main/train_cifar_apart.py --dataset cifar10/cifar100 --model wideresnet40_2/preact_resnet \
--ratio 1 --epsilon 0.1 --groups 16 \
--rho 0.1
```

#### Tiny-ImageNet & ImageNet

<br>