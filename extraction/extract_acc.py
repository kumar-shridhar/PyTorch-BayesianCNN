import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import re
import numpy as np
plt.rc('font', family='serif', size=32)
plt.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, 'axes.labelsize': 32})

# change for given number of tasks
os.chdir("/home/felix/Dropbox/Research/publications/Bayesian_CNN/results/")

with open("diagnostics_MNIST.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_1 = acc[0::2]
valid_1 = acc[1::2]

train_1 = np.array(train_1).astype(np.float32)
valid_1 = np.array(valid_1).astype(np.float32)

with open("diagnostics_CIFAR-10.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_2 = acc[0::2]
valid_2 = acc[1::2]

train_2 = np.array(train_2).astype(np.float32)
valid_2 = np.array(valid_2).astype(np.float32)

with open("diagnostics_BBBMNIST.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_3 = acc[0::2]
valid_3 = acc[1::2]

train_3 = np.array(train_3).astype(np.float32)
valid_3 = np.array(valid_3).astype(np.float32)

with open("diagnostics_BBBCIFAR-10.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_4 = acc[0::2]
valid_4 = acc[1::2]

train_4 = np.array(train_4).astype(np.float32)
valid_4 = np.array(valid_4).astype(np.float32)


f = plt.figure(figsize=(20, 16))


print(valid_1)
print(valid_2)
print(valid_3)
print(valid_4)

plt.plot(valid_1, label=r"MNIST, $MLE$", color='maroon', linestyle='--')
plt.plot(valid_2, label=r"CIFAR-10, $MLE$", color='darkblue', linestyle='--')
plt.plot(valid_3, label=r"MNIST, prior: $U(a, b)$", color='maroon')
plt.plot(valid_4, label=r"CIFAR-10, prior: $U(a, b)$", color='darkblue')
plt.plot(valid_5, label=r"Fashion-MNIST, prior: $U(a, b)$", color='m')
plt.plot(valid_6, label=r"Fashion-MNIST, $MLE$", color='m', linestyle='--')



plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
x_ticks = range(len(valid_1))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

plt.legend(fontsize=28)

plt.savefig("results_CNN.png")

