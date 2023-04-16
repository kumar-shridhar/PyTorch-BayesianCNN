from __future__ import print_function
import argparse
from math import log10

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from super_resolution.model import Net, BayesianNet
from super_resolution.data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

def train(model, optimizer, criterion, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        torch.cuda.empty_cache()
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        kl = 0
        if model.__class__.__name__ == 'BayesianNet':
            output, kl = model(input)
        else:
            output = model(input)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch, epoch_loss / len(training_data_loader)


def test(model, criterion):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            if model.__class__.__name__ == 'BayesianNet':
                prediction, kl = model(input)
            else:
                prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(model, epoch, upscale_factor):
    ckpt_dir = f'./super_resolution/models/{model.__class__.__name__}/'
    model_out_path = f"./super_resolution/models/{model.__class__.__name__}/epoch_{epoch}_upscale_factor_{upscale_factor}_device_{str(device).split(':')[0]}.pth"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


for upscale_factor in [opt.upscale_factor]:
    models = [
        BayesianNet(1, upscale_factor, priors).to(device),
        Net(upscale_factor=upscale_factor).to(device)
    ]
    for model in models:
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        epochs = []
        epoch_losses = []
        avg_psnrs = []
        for epoch in range(1, opt.nEpochs + 1):
            # TODO: for stats see train.py
            epoch, epoch_loss = train(model, optimizer, criterion, epoch)
            avg_psnr = test(model, criterion)
            checkpoint(model, epoch, upscale_factor)

            epochs.append(epoch)
            epoch_losses.append(epoch_loss)
            avg_psnrs.append(avg_psnr)
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(epoch_losses, color='orange', label='train loss')
        # plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'./graphs/loss_model_{model.__class__.__name__}_epochs_{opt.nEpochs}_upscale_factor_{upscale_factor}.png')
        # psnr plots
        plt.figure(figsize=(10, 7))
        plt.plot(avg_psnrs, color='green', label='train PSNR dB')
        # plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.savefig(f'./graphs/psnr_model_{model.__class__.__name__}_epochs_{opt.nEpochs}_upscale_factor_{upscale_factor}.png')
