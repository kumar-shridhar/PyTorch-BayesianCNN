############### Configuration file for Bayesian ###############
n_epochs = 30
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256

train_ens = 1
valid_ens = 1
record_mean_var = True
recording_freq_per_epoch = 32

# Cross-module global variables
mean_var_dir = None
record_now = False
epoch_no = None
