import torch.nn as nn
from mixture_layers import MixtureLinear, MixtureConv2d
from layers import FlattenLayer, ModuleWrapper


def _get_individual_model_weights(layer, individual_weights, bias=False):
    """
    layer: string eg. conv1, fc1 etc.
    individual_weights: List
    """
    if bias:
        return ([weights[layer + '.bias_mu'] for weights in individual_weights],
                [weights[layer + '.bias_rho'] for weights in individual_weights])
    return ([weights[layer + '.W_mu'] for weights in individual_weights],
            [weights[layer + '.W_rho'] for weights in individual_weights])


def W_individual(layer, individual_weights):
    return _get_individual_model_weights(layer, individual_weights)


def bias_individual(layer, individual_weights):
    mu, rho = _get_individual_model_weights(layer, individual_weights, bias=True)
    return {'bias_mu_individual': mu,
            'bias_rho_individual': rho}


class MixtureLeNet(ModuleWrapper):
    '''The architecture of LeNet with Mixture Layers'''

    def __init__(self, outputs, inputs, num_tasks, individual_weights, activation_type='softplus'):
        super(MixtureLeNet, self).__init__()

        self.num_classes = outputs

        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = MixtureConv2d(inputs, 6, 5, num_tasks, *W_individual('conv1', individual_weights), 
                                   padding=0, bias=True, **bias_individual('conv1', individual_weights))
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = MixtureConv2d(6, 16, 5, num_tasks, *W_individual('conv2', individual_weights),
                                   padding=0, bias=True, **bias_individual('conv2', individual_weights))
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = MixtureLinear(5 * 5 * 16, 120, num_tasks, *W_individual('fc1', individual_weights),
                                 bias=True, **bias_individual('fc1', individual_weights))
        self.act3 = self.act()

        self.fc2 = MixtureLinear(120, 84, num_tasks, *W_individual('fc2', individual_weights),
                                 bias=True, **bias_individual('fc2', individual_weights))
        self.act4 = self.act()

        self.fc3 = MixtureLinear(84, outputs, num_tasks, *W_individual('fc3', individual_weights),
                                 bias=True, **bias_individual('fc3', individual_weights))
