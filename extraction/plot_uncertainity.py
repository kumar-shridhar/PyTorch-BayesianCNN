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
os.chdir("../")

with open("diagnostics_Bayeslenet_mnist_25_.txt", 'r') as file:
    unc = re.findall(r"'Epistemic Uncertainity:':\s+(.*?)\}", file.read())
#print(unc)

valid_1 = unc[0::1]

valid_1 = np.array(valid_1).astype(np.float32)
print (valid_1)

with open("diagnostics_Bayeslenet_cifar10_25_.txt", 'r') as file:
    unc = re.findall(r"'Epistemic Uncertainity:':\s+(.*?)\}", file.read())
#print(unc)

valid_2 = unc[0::1]

valid_2 = np.array(valid_2).astype(np.float32)
print (valid_2)

f = plt.figure(figsize=(20, 16))

print ("Plot saved in results folder")

plt.plot(valid_1, label=r"Epistemic Uncertainity MNIST", color='maroon')
plt.plot(valid_2, label=r"Epistemic Uncertainity CIFAR-10", color='green')

plt.xlabel("Epochs")
plt.ylabel("Epistemic Uncertainity")
x_ticks = range(len(valid_1))
y_ticks = range(2)
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))
plt.yticks(y_ticks[1::2], map(lambda y: y+1, y_ticks[1::2]))

plt.legend(fontsize=16)

plt.savefig("certainity.png")