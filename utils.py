import os
import torch
import numpy as np

import config_bayesian as cfg


# cifar10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

# check if dimension is correct

# def dimension_check(x, dim=None, keepdim=False):
#     if dim is None:
#         x, dim = x.view(-1), 0

#     return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename, array_type):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    file.write(f"{array_type}${shape}${cfg.curr_batch_no}${cfg.curr_epoch_no}\n")
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()


def load_mean_std_from_file(filename):
    file = open(filename, 'r')
    means = []
    stds = []
    while True:
        desc = file.readline()
        str_array = file.readline()
        if desc=="":
            file.close()
            return means, stds

        array_type, shape, _, _ = desc.strip().split('$')
        shape = np.fromstring(shape, sep=' ', dtype=np.int64)
        numpy_array = np.fromstring(str_array, sep=' ').reshape(shape)
        if array_type=="mean":
            means.append(numpy_array)
        else:
            stds.append(numpy_array)


def get_file_info(filename):
    file = open(filename, 'r')
    file_desc = {
        'layer_name': None,
        'batch_size': None,
        'original_shape': None,
        'number_of_nodes': None,
        'number_of_epochs': None,
        'recording_frequency_per_epoch': None
    }

    path_desc = filename.split('/')
    file_desc['layer_name'] = path_desc[-1][:-4]

    shape = file.readline().strip().split('$')[1]
    shape = np.fromstring(shape, sep=' ', dtype=np.int64)
    file_desc['original_shape'] = shape
    file_desc['batch_size'] = shape[0]
    file_desc['number_of_nodes'] = np.prod(shape[1:])

    # go to beginning of the file
    file.seek(0)
    freq_per_epoch = 0
    while True:
        desc = file.readline()
        str_array = file.readline()
        if desc=="":
            break
        _, _, _, epoch_no = desc.strip().split('$')
        epoch_no = int(epoch_no)
        if epoch_no==0:
            freq_per_epoch += 1
        else:
            break
    file_desc['recording_frequency_per_epoch'] = freq_per_epoch // 2

    # go to ending of the file
    file.seek(0, os.SEEK_END)
    end = file.tell()
    i = 0
    while True:
        file.seek(end-i)
        char = file.read(1)
        if char == '$':
            break
        i += 1
    epoch = file.readline().strip()
    file_desc['number_of_epochs'] = int(epoch) + 1 # since epochs start from 0

    return file_desc
