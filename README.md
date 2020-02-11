# Bayesian CNN with Variational Inference for Image Recognition task

We introduce **Bayesian convolutional neural networks with variational inference**, a variant of convolutional neural networks (CNNs), in which the intractable posterior probability distributions over weights are inferred by **Bayes by Backprop**. We demonstrate how our proposed variational inference method achieves performances equivalent to frequentist inference in identical architectures on several datasets (MNIST, CIFAR10, CIFAR100) as described in the [paper](https://arxiv.org/abs/1901.02731).

---------------------------------------------------------------------------------------------------------


### Filter weight distributions in a Bayesian Vs Frequentist approach

![Distribution over weights in a CNN's filter.](experiments/figures/BayesCNNwithdist.png)

---------------------------------------------------------------------------------------------------------

### Fully Bayesian perspective of an entire CNN 

![Distributions must be over weights in convolutional layers and weights in fully-connected layers.](experiments/figures/CNNwithdist_git.png)

---------------------------------------------------------------------------------------------------------



### Make your custom Bayesian Network?
To make a custom Bayesian Network, inherit `layers.misc.ModuleWrapper` instead of `torch.nn.Module` and use `layers.BBBLinear.BBBLinear` and `layers.BBBConv.BBBConv2d` instead of `torch.nn.Conv2d` and `torch.nn.Linear`. Moreover, no don't need to define `forward` method. It'll be automatically be taken care of. 

For example:  
```python
class Net(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 16, 5, strides=2)
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(800, 10)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    x = x.view(-1, 800)
    x = self.fc(x)
    return x
```
Above Network can be converted to Bayesian as follows:
```python
class Net(ModuleWrapper):

  def __init__(self):
    super().__init__()
    self.conv = BBBConv2d(3, 16, 5, strides=2, alpha_shape=(1,1))
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU()
    self.flatten = FlattenLayer(800)
    self.fc = BBBLinear(800, 10, alpha_shape=(1,1))
```

#### Notes: 
1. Add `FlattenLayer` before first `BBBLinear` block.  
2. `forward` method of the model will return a tuple as `(logits, kl)`.

---------------------------------------------------------------------------------------------------------

### How to perform standard experiments?
Currently, following datasets and models are supported.  
* Datasets: MNIST, CIFAR10, CIFAR100  
* Models: AlexNet, LeNet, 3Conv3FC  

#### Bayesian

`python main_bayesian.py`
* set hyperparameters in `config_bayesian.py`


#### Frequentist

`python main_frequentist.py`
* set hyperparameters in `config_frequentist.py`

---------------------------------------------------------------------------------------------------------



### Directory Structure:
`layers/`:  Contains ModuleWrapper, FlattenLayer, Bayesian layers (BBBConv2d and BBBLinear).  
`models/BayesianModels/`: Contains standard Bayesian models (BBBLeNet, BBBAlexNet, BBB3Conv3FC).  
`models/NonBayesianModels/`: Contains standard Non-Bayesian models (LeNet, AlexNet).  
`checkpoints/`: Checkpoint directory for the best model will be saved here.  
`tests/`: Basic unittest cases for layers and models.  
`main_bayesian.py`: Train and Evaluate Bayesian models.  
`config_bayesian.py`: Hyperparameters for `main_bayesian` file.  
`main_frequentist.py`: Train and Evaluate non-Bayesian (Frequentist) models.  
`config_frequentist.py`: Hyperparameters for `main_frequentist` file. 

---------------------------------------------------------------------------------------------------------





If you are using this work, please cite:

```
@article{shridhar2019comprehensive,
  title={A comprehensive guide to bayesian convolutional neural network with variational inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1901.02731},
  year={2019}
}
```

--------------------------------------------------------------------------------------------------------
