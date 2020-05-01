############### Configuration file for Bayesian ###############
n_epochs = 10
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 10
valid_ens = 5
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value

record_mean_var = False
recording_freq_per_epoch = 32
record_layers = ['fc3']

# Cross-module global variables
mean_var_dir = None
record_now = False
curr_epoch_no = None
curr_batch_no = None
