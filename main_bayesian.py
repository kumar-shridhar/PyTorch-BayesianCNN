from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility — so accuracy changes you measure come from your edits,
# not run-to-run noise
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if net_type == 'lenet':
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == 'alexnet':
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif net_type == '3conv3fc':
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError(
            'Network should be either [LeNet / AlexNet / 3Conv3FC]'
        )


def train_model(
    net,
    optimizer,
    criterion,
    trainloader,
    num_ens=1,
    beta_type=0.1,
    epoch=None,
    num_epochs=None,
    grad_clip_norm=5.0,
):
    net.train()

    training_loss = 0.0
    accs = []
    kl_list = []

    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = torch.zeros(
            inputs.shape[0],
            net.num_classes,
            num_ens
        ).to(device)

        kl = 0.0

        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / num_ens
        kl_list.append(kl.item())

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(
            i - 1,
            len(trainloader),
            beta_type,
            epoch,
            num_epochs
        )

        loss = criterion(log_outputs, labels, kl, beta)

        loss.backward()

        # Gradient clipping — BBB layers sample noise every forward pass,
        # which makes gradients noisier than a plain CNN; clipping
        # stabilizes training and helps final accuracy.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.item()

    return (
        training_loss / len(trainloader),
        np.mean(accs),
        np.mean(kl_list)
    )


def validate_model(
    net,
    criterion,
    validloader,
    num_ens=1,
    beta_type=0.1,
    epoch=None,
    num_epochs=None
):
    """Calculate ensemble accuracy and NLL Loss"""

    net.eval()

    valid_loss = 0.0
    accs = []

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(validloader):

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = torch.zeros(
                inputs.shape[0],
                net.num_classes,
                num_ens
            ).to(device)

            kl = 0.0

            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1)

            kl = kl / num_ens

            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(
                i - 1,
                len(validloader),
                beta_type,
                epoch,
                num_epochs
            )

            valid_loss += criterion(
                log_outputs,
                labels,
                kl,
                beta
            ).item()

            accs.append(metrics.acc(log_outputs, labels))

    return valid_loss / len(validloader), np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)

    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset,
        testset,
        valid_size,
        batch_size,
        num_workers
    )

    net = getModel(
        net_type,
        inputs,
        outputs,
        priors,
        layer_type,
        activation_type
    ).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = (
        f'checkpoints/{dataset}/bayesian/'
        f'model_{net_type}_{layer_type}_{activation_type}.pt'
    )

    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)

    # weight_decay added — the KL term already regularizes toward the
    # prior, so keep this small relative to a plain CNN.
    optimizer = Adam(net.parameters(), lr=lr_start, weight_decay=1e-5)

    lr_sched = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=6
    )

    # Track best VALIDATION ACCURACY, not lowest validation loss.
    # valid_loss = NLL + beta * KL, and beta changes across epochs
    # (Blundell schedule), so it isn't a stable, comparable quantity
    # epoch-to-epoch. Accuracy is what we actually care about.
    best_valid_acc = 0.0

    for epoch in range(n_epochs):

        train_loss, train_acc, train_kl = train_model(
            net,
            optimizer,
            criterion,
            train_loader,
            num_ens=train_ens,
            beta_type=beta_type,
            epoch=epoch,
            num_epochs=n_epochs
        )

        # More MC samples at validation time = lower-variance, generally
        # more accurate estimate of the model's true predictive
        # performance, at the cost of extra forward passes (no retraining
        # needed). Bump this if you can afford the compute.
        eval_ens = max(valid_ens, 10)

        valid_loss, valid_acc = validate_model(
            net,
            criterion,
            valid_loader,
            num_ens=eval_ens,
            beta_type=beta_type,
            epoch=epoch,
            num_epochs=n_epochs
        )

        old_lr = optimizer.param_groups[0]['lr']

        lr_sched.step(valid_loss)

        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(
                f"Learning rate reduced from "
                f"{old_lr:.6e} to {new_lr:.6e}"
            )

        print(
            'Epoch: {} \tTraining Loss: {:.4f} '
            '\tTraining Accuracy: {:.4f} '
            '\tValidation Loss: {:.4f} '
            '\tValidation Accuracy: {:.4f} '
            '\ttrain_kl_div: {:.4f}'.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                train_kl
            )
        )

        if valid_acc >= best_valid_acc:
            print(
                'Validation accuracy increased '
                '({:.4f} --> {:.4f}). Saving model ...'.format(
                    best_valid_acc,
                    valid_acc
                )
            )

            torch.save(net.state_dict(), ckpt_name)
            best_valid_acc = valid_acc

    # Final honest number: reload the best checkpoint and evaluate on the
    # held-out test set (previously the script never touched test_loader,
    # so you had no unbiased accuracy figure at all).
    net.load_state_dict(torch.load(ckpt_name, map_location=device))
    test_loss, test_acc = validate_model(
        net,
        criterion,
        test_loader,
        num_ens=max(valid_ens, 20),
        beta_type=beta_type,
        epoch=n_epochs,
        num_epochs=n_epochs
    )
    print(
        'Best Validation Accuracy: {:.4f}'
        '\tTest Loss: {:.4f} \tTest Accuracy: {:.4f}'.format(
            best_valid_acc, test_loss, test_acc
        )
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch Bayesian Model Training"
    )

    parser.add_argument(
        '--net_type',
        default='lenet',
        type=str,
        help='model'
    )

    parser.add_argument(
        '--dataset',
        default='MNIST',
        type=str,
        help='dataset = [MNIST/CIFAR10/CIFAR100]'
    )

    args = parser.parse_args()

    run(args.dataset, args.net_type)