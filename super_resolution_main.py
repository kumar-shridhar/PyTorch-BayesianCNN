from __future__ import print_function
import argparse
import csv
from math import log10

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
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
# if not opt.mps and torch.backends.mps.is_available():
#     raise Exception("Found mps device, please run with --mps to enable macOS GPU")

torch.manual_seed(opt.seed)
use_mps = opt.mps and torch.backends.mps.is_available()

if opt.cuda:
    device = torch.device("cuda")
elif use_mps:
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
model = BayesianNet(1, opt.upscale_factor, priors).to(device)
# model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

complete_avg_loss_list = dict()
complete_avg_psnr_list = dict()

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        torch.cuda.empty_cache()
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output, kl = model(input)
        # Should be: [batchSize, 1, 256, 256]
        # print(output.shape)
        # print(target.shape)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    
    complete_avg_loss = epoch_loss / len(training_data_loader)
    complete_avg_loss_list[f'Epoch: {epoch}'] = complete_avg_loss
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, complete_avg_loss))


def test(epoch):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction, kl = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    complete_avg_psnr = avg_psnr / len(testing_data_loader)
    complete_avg_psnr_list[f'Epoch: {epoch}'] = complete_avg_psnr
    
    print("===> Avg. PSNR: {:.4f} dB".format(complete_avg_psnr))


def checkpoint(epoch):
    model_out_path = "./super_resolution/models/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test(epoch)
    checkpoint(epoch)
print(complete_avg_loss_list)
print(complete_avg_psnr_list)

with open('complete_avg_loss.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in complete_avg_loss_list.items():
       writer.writerow([key, value])
       
with open('complete_avg_psnr.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in complete_avg_psnr_list.items():
       writer.writerow([key, value])