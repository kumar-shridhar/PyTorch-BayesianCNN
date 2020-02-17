############### Configuration file for Bayesian ###############
n_epochs = 10
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256

train_ens = 1
valid_ens = 1
record_mean_var = True

# Cross-module global variables
net_type = None
dataset = None
mean_var_dir = None