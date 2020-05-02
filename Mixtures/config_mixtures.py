############### Configuration file for Training of SplitMNIST and Mixtures ###############
layer_type = 'lrt'
activation_type = 'softplus'

n_epochs = 3
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 10
valid_ens = 5
