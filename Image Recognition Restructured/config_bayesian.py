############### Configuration file for Bayesian ###############
n_epochs = 150
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256

train_ens = 1

# Logger format
logger_fmt = {'kl': '3.3e',
       'tr_loss': '3.3e',
       'tr_acc': '.4f',
       'zero_mean_tr_loss': '3.3e',
       'zero_mean_tr_acc': '.4f',
       'te_acc_ens100': '.4f',
       'te_acc_stoch': '.4f',
       'te_acc_ens10': '.4f',
       'te_acc_perm_sigma': '.4f',
       'te_acc_zero_mean': '.4f',
       'te_acc_perm_sigma_ens': '.4f',
       'te_acc_zero_mean_ens': '.4f',
       'te_nll_ens100': '.4f',
       'te_nll_stoch': '.4f',
       'te_nll_ens10': '.4f',
       'te_nll_perm_sigma': '.4f',
       'te_nll_zero_mean': '.4f',
       'te_nll_perm_sigma_ens': '.4f',
       'te_nll_zero_mean_ens': '.4f',
       'time': '.3f'}
