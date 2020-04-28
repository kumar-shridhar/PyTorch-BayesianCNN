import sys
sys.path.append('..')

import os
import datetime
import torch
import contextlib

from utils_mixture import *
from layers.BBBLinear import BBBLinear


@contextlib.contextmanager
def print_to_logfile(file):
    # capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    sys.stdout = logger
    try:
        yield logger.log
    finally:
        sys.stdout = _stdout


def initiate_experiment(experiment):

    def decorator(*args, **kwargs):
        log_file_dir = "experiments/mixtures/"
        log_file = log_file_dir + experiment.__name__ + ".txt"
        if not os.path.exists(log_file):
            os.makedirs(log_file_dir, exist_ok=True)
        with print_to_logfile(open(log_file, 'a')):
            print("Performing experiment:", experiment.__name__)
            print("Date-Time:", datetime.datetime.now())
            print("\n", end="")
            print("Args:", args)
            print("Kwargs:", kwargs)
            print("\n", end="")
            experiment(*args, **kwargs)
            print("\n\n", end="")
    return decorator


@initiate_experiment
def experiment_regular_prediction_bayesian(weights_dir=None, num_ens=10):
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/2-tasks/" if weights_dir is None else weights_dir

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, bayesian=True, pretrained=True, weights_dir=weights_dir)
    net1.cuda()
    net2.cuda()

    print("Model-1, Task-1-Dataset=> Accuracy:", predict_regular(net1, loaders1[1], bayesian=True, num_ens=num_ens))
    print("Model-2, Task-2-Dataset=> Accuracy:", predict_regular(net2, loaders2[1], bayesian=True, num_ens=num_ens))


@initiate_experiment
def experiment_regular_prediction_frequentist(weights_dir=None):
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/frequentist/splitted/2-tasks/" if weights_dir is None else weights_dir

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, bayesian=False, pretrained=True, weights_dir=weights_dir)
    net1.cuda()
    net2.cuda()

    print("Model-1, Task-1-Dataset=> Accuracy:", predict_regular(net1, loaders1[1], bayesian=False))
    print("Model-2, Task-2-Dataset=> Accuracy:", predict_regular(net2, loaders2[1], bayesian=False))


@initiate_experiment
def experiment_simultaneous_without_mixture_model_with_uncertainty(uncertainty_type="epistemic_softmax", T=25, weights_dir=None):
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/2-tasks/" if weights_dir is None else weights_dir

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, True, True, weights_dir)
    net1.cuda()
    net2.cuda()

    print("Both Models, Task-1-Dataset=> Accuracy: {:.3}\tModel-1-Preferred: {:.3}\tModel-2-Preferred: {:.3}\t" \
          "Task-1-Dataset-Uncertainty: {:.3}\tTask-2-Dataset-Uncertainty: {:.3}".format(
        *predict_using_uncertainty_separate_models(net1, net2, loaders1[1], uncertainty_type=uncertainty_type, T=T)))
    print("Both Models, Task-2-Dataset=> Accuracy: {:.3}\tModel-1-Preferred: {:.3}\tModel-2-Preferred: {:.3}\t" \
          "Task-1-Dataset-Uncertainty: {:.3}\tTask-2-Dataset-Uncertainty: {:.3}".format(
        *predict_using_uncertainty_separate_models(net1, net2, loaders2[1], uncertainty_type=uncertainty_type, T=T)))


@initiate_experiment
def experiment_simultaneous_without_mixture_model_with_confidence(weights_dir=None):
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/frequentist/splitted/2-tasks/" if weights_dir is None else weights_dir

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, False, True, weights_dir)
    net1.cuda()
    net2.cuda()

    print("Both Models, Task-1-Dataset=> Accuracy: {:.3}\tModel-1-Preferred: {:.3}\tModel-2-Preferred: {:.3}".format(
        *predict_using_confidence_separate_models(net1, net2, loaders1[1])))
    print("Both Models, Task-2-Dataset=> Accuracy: {:.3}\tModel-1-Preferred: {:.3}\tModel-2-Preferred: {:.3}".format(
        *predict_using_confidence_separate_models(net1, net2, loaders2[1])))


@initiate_experiment
def wip_experiment_average_weights_mixture_model():
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/2-tasks/"

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, True, weights_dir)
    net1.cuda()
    net2.cuda()
    net_mix = get_mixture_model(num_tasks, weights_dir, include_last_layer=True)
    net_mix.cuda()

    print("Model-1, Loader-1:", calculate_accuracy(net1, loaders1[1]))
    print("Model-2, Loader-2:", calculate_accuracy(net2, loaders2[1]))
    print("Model-1, Loader-2:", calculate_accuracy(net1, loaders2[1]))
    print("Model-2, Loader-1:", calculate_accuracy(net2, loaders1[1]))
    print("Model-Mix, Loader-1:", calculate_accuracy(net_mix, loaders1[1]))
    print("Model-Mix, Loader-2:", calculate_accuracy(net_mix, loaders2[1]))


@initiate_experiment
def wip_experiment_simultaneous_average_weights_mixture_model_with_uncertainty():
    num_tasks = 2
    weights_dir = "checkpoints/MNIST/bayesian/splitted/2-tasks/"

    loaders1, loaders2 = get_splitmnist_dataloaders(num_tasks)
    net1, net2 = get_splitmnist_models(num_tasks, True, weights_dir)
    net1.cuda()
    net2.cuda()
    net_mix = get_mixture_model(num_tasks, weights_dir, include_last_layer=False)
    net_mix.cuda()

    # Creating 2 sets of last layer
    fc3_1 = BBBLinear(84, 5, name='fc3_1') # hardcoded for lenet
    weights_1 = torch.load(weights_dir + "model_lenet_2.1.pt")
    fc3_1.W = torch.nn.Parameter(weights_1['fc3.W'])
    fc3_1.log_alpha = torch.nn.Parameter(weights_1['fc3.log_alpha'])

    fc3_2 = BBBLinear(84, 5, name='fc3_2') # hardcoded for lenet
    weights_2 = torch.load(weights_dir + "model_lenet_2.2.pt")
    fc3_2.W = torch.nn.Parameter(weights_2['fc3.W'])
    fc3_2.log_alpha = torch.nn.Parameter(weights_2['fc3.log_alpha'])

    fc3_1, fc3_2 = fc3_1.cuda(), fc3_2.cuda()

    print("Model-1, Loader-1:", calculate_accuracy(net1, loaders1[1]))
    print("Model-2, Loader-2:", calculate_accuracy(net2, loaders2[1]))
    print("Model-Mix, Loader-1:", predict_using_epistemic_uncertainty_with_mixture_model(net_mix, fc3_1, fc3_2, loaders1[1]))
    print("Model-Mix, Loader-2:", predict_using_epistemic_uncertainty_with_mixture_model(net_mix, fc3_1, fc3_2, loaders2[1]))


if __name__ == '__main__':
    experiment_simultaneous_without_mixture_model_with_confidence()
