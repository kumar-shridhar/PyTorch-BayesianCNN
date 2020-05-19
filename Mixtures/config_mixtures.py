############### Configuration file for Training of SplitMNIST and Mixtures ###############
layer_type = 'lrt'
activation_type = 'softplus'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 5
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 10
valid_ens = 5
