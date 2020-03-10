import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import data
from main_bayesian import getModel


mnist_set = None
notmnist_set = None

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])


def init_dataset(notmnist_dir):
    global mnist_set
    global notmnist_set
    mnist_set, _, _, _ = data.getDataset('MNIST')
    notmnist_set = torchvision.datasets.ImageFolder(root=notmnist_dir)


def calc_uncertainty_softmax(model, input_image, T=15, return_pred=False, return_diag=False):
    input_image = input_image.unsqueeze(0)
    p_hat = []
    if return_pred:
        preds = []
    for t in range(T):
        net_out, _ = model(input_image)
        if return_pred:
            preds.append(net_out)
        p_hat.append(F.softmax(net_out, dim=1).cpu().detach())

    p_hat = torch.cat(p_hat, dim=0).numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)

    if return_diag:
        epistemic, aleatoric = np.diag(epistemic), np.diag(aleatoric)

    # Prediction
    if return_pred:
        preds = torch.cat(preds, dim=0).detach().numpy()
        pred = np.sum(preds, axis=0).squeeze()
        return epistemic, aleatoric, np.argmax(pred)

    return epistemic, aleatoric


def calc_uncertainty_normalized(model, input_image, T=15, return_pred=False, return_diag=False):
    input_image = input_image.unsqueeze(0)
    p_hat = []
    if return_pred:
        preds = []
    for t in range(T):
        net_out, _ = model(input_image)
        if return_pred:
            preds.append(net_out)
        prediction = F.softplus(net_out)
        prediction = prediction / torch.sum(prediction, dim=1)
        p_hat.append(prediction.cpu().detach())

    p_hat = torch.cat(p_hat, dim=0).numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)

    if return_diag:
        epistemic, aleatoric = np.diag(epistemic), np.diag(aleatoric)

    # Prediction
    if return_pred:
        preds = torch.cat(preds, dim=0).detach().numpy()
        pred = np.sum(preds, axis=0).squeeze()
        return epistemic, aleatoric, np.argmax(pred)

    return epistemic, aleatoric


def get_sample(dataset, sample_type='mnist'):
    idx = np.random.randint(len(dataset.targets))
    if sample_type=='mnist':
        sample = dataset.data[idx]
        truth = dataset.targets[idx]
    else:
        path, truth = dataset.samples[idx]
        sample = torch.from_numpy(np.array(Image.open(path)))

    sample = sample.unsqueeze(0)
    sample = transform(sample)
    return sample, truth


def run(net_type, weight_path, notmnist_dir):
    init_dataset(notmnist_dir)

    net = getModel(net_type, 1, 10)
    net.load_state_dict(torch.load(weight_path))
    net.train()

    fig = plt.figure()
    fig.suptitle('Uncertainty Estimation', fontsize='x-large')
    mnist_img = fig.add_subplot(321)
    notmnist_img = fig.add_subplot(322)
    epi_stats_norm = fig.add_subplot(323)
    ale_stats_norm = fig.add_subplot(324)
    epi_stats_soft = fig.add_subplot(325)
    ale_stats_soft = fig.add_subplot(326)

    sample_mnist, truth_mnist = get_sample(mnist_set)
    epi_mnist_norm, ale_mnist_norm, pred_mnist = calc_uncertainty_normalized(net, sample_mnist, return_pred=True, return_diag=True)
    epi_mnist_soft, ale_mnist_soft, pred_mnist = calc_uncertainty_softmax(net, sample_mnist, return_pred=True, return_diag=True)
    mnist_img.imshow(sample_mnist.squeeze(), cmap='gray')
    mnist_img.axis('off')
    mnist_img.set_title('MNIST Truth: {} Prediction: {}'.format(int(truth_mnist), int(pred_mnist)))

    sample_notmnist, truth_notmnist = get_sample(notmnist_set, sample_type='notmnist')
    epi_notmnist_norm, ale_notmnist_norm, pred_notmnist = calc_uncertainty_normalized(net, sample_notmnist, return_pred=True, return_diag=True)
    epi_notmnist_soft, ale_notmnist_soft, pred_notmnist = calc_uncertainty_softmax(net, sample_notmnist, return_pred=True, return_diag=True)
    notmnist_img.imshow(sample_notmnist.squeeze(), cmap='gray')
    notmnist_img.axis('off')
    notmnist_img.set_title('notMNIST Truth: {}({}) Prediction: {}({})'.format(
        int(truth_notmnist), chr(65 + truth_notmnist), int(pred_notmnist), chr(65 + pred_notmnist)))

    x = list(range(10))
    data = pd.DataFrame({
        'epistemic_norm': np.hstack([epi_mnist_norm, epi_notmnist_norm]),
        'aleatoric_norm': np.hstack([ale_mnist_norm, ale_notmnist_norm]),
        'epistemic_soft': np.hstack([epi_mnist_soft, epi_notmnist_soft]),
        'aleatoric_soft': np.hstack([ale_mnist_soft, ale_notmnist_soft]),
        'category': np.hstack([x, x]),
        'dataset': np.hstack([['MNIST']*10, ['notMNIST']*10])
    })
    print(data)
    sns.barplot(x='category', y='epistemic_norm', hue='dataset', data=data, ax=epi_stats_norm)
    sns.barplot(x='category', y='aleatoric_norm', hue='dataset', data=data, ax=ale_stats_norm)
    epi_stats_norm.set_title('Epistemic Uncertainty (Normalized)')
    ale_stats_norm.set_title('Aleatoric Uncertainty (Normalized)')

    sns.barplot(x='category', y='epistemic_soft', hue='dataset', data=data, ax=epi_stats_soft)
    sns.barplot(x='category', y='aleatoric_soft', hue='dataset', data=data, ax=ale_stats_soft)
    epi_stats_soft.set_title('Epistemic Uncertainty (Softmax)')
    ale_stats_soft.set_title('Aleatoric Uncertainty (Softmax)')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Uncertainty Estimation b/w MNIST and notMNIST")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--weights_path', default='checkpoints/MNIST/bayesian/model_lenet.pt', type=str, help='weights for model')
    parser.add_argument('--notmnist_dir', default='data/notMNIST_small/', type=str, help='weights for model')
    args = parser.parse_args()

    run(args.net_type, args.weights_path, args.notmnist_dir)
