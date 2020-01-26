import os
import sys
import pytest
import unittest
import numpy as np
import torch
import torch.nn as nn

import utils
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC


cuda_available = torch.cuda.is_available()
bayesian_models = [BBBLeNet, BBBAlexNet, BBB3Conv3FC]
non_bayesian_models = [LeNet, AlexNet, ThreeConvThreeFC]

class TestModelForwardpass:

    @pytest.mark.parametrize("model", bayesian_models)
    def test_bayesian_cpu(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        out = net.probforward(batch)
        assert out[0].shape[0]==batch_size

    @pytest.mark.parametrize("model", non_bayesian_models)
    def test_non_bayesian_cpu(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        out = net(batch)
        assert out.shape[0]==batch_size

    @pytest.mark.skipif(not cuda_available, reason="CUDA not available")
    @pytest.mark.parametrize("model", bayesian_models)
    def test_bayesian_gpu(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        if cuda_available:
            net = net.cuda()
            batch = batch.cuda()
        out = net.probforward(batch)
        assert out[0].shape[0]==batch_size

    @pytest.mark.skipif(not cuda_available, reason="CUDA not available")
    @pytest.mark.parametrize("model", non_bayesian_models)
    def test_non_bayesian_gpu(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        if cuda_available:
            net = net.cuda()
            batch = batch.cuda()
        out = net(batch)
        assert out.shape[0]==batch_size

