import os
import sys
import math
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
from layers.BBBConv import BBBConv2d
from layers.BBBLinear import  BBBLinear
from layers.misc import FlattenLayer

cuda_available = torch.cuda.is_available()
bayesian_models = [BBBLeNet, BBBAlexNet, BBB3Conv3FC]
non_bayesian_models = [LeNet, AlexNet, ThreeConvThreeFC]

class TestModelForwardpass:

    @pytest.mark.parametrize("model", bayesian_models)
    def test_cpu_bayesian(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        out = net(batch)
        assert out[0].shape[0]==batch_size
    
    @pytest.mark.parametrize("model", non_bayesian_models)
    def test_cpu_frequentist(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        out = net(batch)
        assert out.shape[0]==batch_size

    @pytest.mark.skipif(not cuda_available, reason="CUDA not available")
    @pytest.mark.parametrize("model", bayesian_models)
    def test_gpu_bayesian(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        if cuda_available:
            net = net.cuda()
            batch = batch.cuda()
        out = net(batch)
        assert out[0].shape[0]==batch_size

    @pytest.mark.skipif(not cuda_available, reason="CUDA not available")
    @pytest.mark.parametrize("model", non_bayesian_models)
    def test_gpu_frequentist(self, model):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 3, 32, 32))
        net = model(10, 3)
        if cuda_available:
            net = net.cuda()
            batch = batch.cuda()
        out = net(batch)
        assert out.shape[0]==batch_size


class TestBayesianLayers:

    def test_flatten(self):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 64, 4, 4))

        layer = FlattenLayer(4 * 4 * 64)
        batch = layer(batch)

        assert batch.shape[0]==batch_size
        assert batch.shape[1]==(4 * 4 *64)

    def test_conv(self):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 16, 24, 24))

        layer = BBBConv2d(16, 6, 4, alpha_shape=(1,1), padding=0, bias=False)
        batch = layer(batch)

        assert batch.shape[0]==batch_size
        assert batch.shape[1]==6

    def test_linear(self):
        batch_size = np.random.randint(1, 256)
        batch = torch.randn((batch_size, 128))

        layer = BBBLinear(128, 64, alpha_shape=(1,1), bias=False)
        batch = layer(batch)

        assert batch.shape[0]==batch_size
        assert batch.shape[1]==64
