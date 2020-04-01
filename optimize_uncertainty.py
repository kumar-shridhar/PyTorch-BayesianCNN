import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import data
from main_bayesian import getModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_time(func): 
    def decorator(*args, **kwargs): 
        begin = time.time() 
        func(*args, **kwargs) 
        end = time.time() 
        print("Time taken in : ", func.__name__, end - begin) 
    return decorator


@calculate_time
def calc_uncertainty_normalized_per_image(model, input_image, T=15):
    input_image = input_image.unsqueeze(0)
    p_hat = []
    for t in range(T):
        net_out, _ = model(input_image)
        prediction = F.softplus(net_out)
        prediction = prediction / torch.sum(prediction, dim=1)
        p_hat.append(prediction.cpu().detach())

    p_hat = torch.cat(p_hat, dim=0).numpy()
    print(p_hat.shape)
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    print(np.sum(np.diag(epistemic)))


@calculate_time
def calc_uncertainty_normalized_per_batch(model, batch, T=15):
    # loop over a batch
    total_epistemic =  0.0
    for i in range(batch.shape[0]):
        input_image = batch[i].unsqueeze(0).cuda()
        p_hat = []
        for t in range(T):
            net_out, _ = model(input_image)
            prediction = F.softplus(net_out)
            prediction = prediction / torch.sum(prediction, dim=1)
            p_hat.append(prediction.cpu().detach())

        p_hat = torch.cat(p_hat, dim=0).numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        total_epistemic += np.sum(np.diag(epistemic))
    print(total_epistemic)


@calculate_time
def calc_uncertainty_normalized_optimized_per_image(model, input_image, T=15):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out, _ = model(input_images)
    prediction = F.softplus(net_out)
    p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    print(np.sum(np.diag(epistemic)))


@calculate_time
def calc_uncertainty_normalized_optimized_per_batch(model, batch, T=15):
    total_epistemic = 0.0
    batch_predictions = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    
    for i in range(T):  # for T batches
        net_out, _ = model(batches[i].cuda())
        prediction = F.softplus(net_out)
        prediction = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
        batch_predictions.append(prediction)
    
    for sample in range(batch.shape[0]):
        # for each sample in a batch
        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        total_epistemic += np.sum(np.diag(epistemic))
    print(total_epistemic)


if __name__ == '__main__':
    trainset, testset, inputs, outputs = data.getDataset('MNIST')
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, 0.2, 256, 4)
    net = getModel('lenet', inputs, outputs).to(device)
    for data, targets in valid_loader:
        print("Uncertainty:", calc_uncertainty_normalized_per_batch(net, data, T=20))
        print("Uncertainty:", calc_uncertainty_normalized_optimized_per_batch(net, data, T=20))
