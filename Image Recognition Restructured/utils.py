import torch
from torch.autograd import Variable
import torch.nn.functional as F
import metrics
import numpy as np


# cifar10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_beta(batch_idx, m, beta_type):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1) 
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m 
    else:
        beta = 0
    return beta


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, xt, xf):
    ret = torch.zeros_like(xt)
    ret[cond] = xt[cond]
    ret[cond ^ 1] = xf[cond ^ 1]
    return ret


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = xm + torch.log(torch.mean(torch.exp(x - xm), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(net, dataloader, num_ens=1):
    """Calculate ensemble accuracy and NLL"""
    accs = []
    nlls = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).cuda()
        for j in range(num_ens):
            outputs[:, :, j] = F.log_softmax(net(inputs), dim=1).data
        accs.append(metrics.logit2acc(logmeanexp(outputs, dim=2), labels))
        nlls.append(F.nll_loss(Variable(logmeanexp(outputs, dim=2)), labels, size_average=False).data.cpu().numpy())
    return np.mean(accs), np.sum(nlls)
