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
os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_MCVI/results/")

with open("diagnostics_blundell.txt", 'r') as file:
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file.read())))]
print(accuracies)

train_1 = [acc[0::2] for acc in accuracies]
valid_1 = [acc[1::2] for acc in accuracies]

train_1 = np.array(train_1).astype(np.float32)[:,49:]
valid_1 = np.array(valid_1).astype(np.float32)[:,49:]

with open("diagnostics_none.txt", 'r') as file:
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file.read())))]
    print(accuracies)

train_2 = [acc[0::2] for acc in accuracies]
valid_2 = [acc[1::2] for acc in accuracies]

train_2 = np.array(train_2).astype(np.float32)
valid_2 = np.array(valid_2).astype(np.float32)

with open("diagnostics_graves.txt", 'r') as file:
    accuracies = [list(map(lambda x: x.split(" ")[-1], re.findall(r"(\'acc\': \d.\d+)", file.read())))]
    print(accuracies)

train_3 = [acc[0::2] for acc in accuracies]
valid_3 = [acc[1::2] for acc in accuracies]

train_3 = np.array(train_3).astype(np.float32)
valid_3 = np.array(valid_3).astype(np.float32)

f = plt.figure(figsize=(20, 16))


print(valid_1)
print(valid_2)
print(valid_3)

plt.plot(valid_1, "--", label=r"$\beta=\frac{2^{M-i}}{2^M-1}$", color='maroon')
plt.plot(valid_2, "--", label=r"$\beta=0$", color='darkblue')
plt.plot(valid_3, "--", label=r"$\beta=\frac{1}{M}$", color='peru')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid_1))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

plt.legend(fontsize=28)

plt.savefig("results_CNN.png")

