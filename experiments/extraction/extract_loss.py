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
os.chdir("../results/")
with open("lenet/diagnostics_NonBayeslenet_cifar10.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_1 = new_loss[0::2]
valid_1 = new_loss[1::2]

train_1 = np.array(train_1).astype(np.float32)
valid_1 = np.array(valid_1).astype(np.float32)

with open("lenet/diagnostics_NonBayeslenet_cifar100.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_2 = new_loss[0::2]
valid_2 = new_loss[1::2]

train_2 = np.array(train_2).astype(np.float32)
valid_2 = np.array(valid_2).astype(np.float32)

with open("lenet/diagnostics_NonBayeslenet_mnist.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_3 = new_loss[0::2]
valid_3 = new_loss[1::2]

train_3 = np.array(train_3).astype(np.float32)
valid_3 = np.array(valid_3).astype(np.float32)

with open("lenet/diagnostics_NonBayeslenet_stl10.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_4 = new_loss[0::2]
valid_4 = new_loss[1::2]

train_4 = np.array(train_4).astype(np.float32)
valid_4 = np.array(valid_4).astype(np.float32)

with open("lenet/diagnostics_Bayeslenet_cifar10.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_5 = new_loss[0::2]
valid_5 = new_loss[1::2]

train_5 = np.array(train_5).astype(np.float32)
valid_5 = np.array(valid_5).astype(np.float32)

with open("lenet/diagnostics_Bayeslenet_cifar100.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)


train_6 = new_loss[0::2]
valid_6 = new_loss[1::2]

train_6 = np.array(train_6).astype(np.float32)
valid_6 = np.array(valid_6).astype(np.float32)

with open("lenet/diagnostics_Bayeslenet_mnist.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_7 = new_loss[0::2]
valid_7 = new_loss[1::2]

train_7 = np.array(train_7).astype(np.float32)
valid_7 = np.array(valid_7).astype(np.float32)

with open("lenet/diagnostics_Bayeslenet_stl10.txt", 'r') as file:
    loss = re.findall(r"'Loss':\s+tensor\((.*?)\)", file.read())
    new_loss = []
    for i in loss:
        loss_new = i[:4]
        new_loss.append(loss_new)
print(new_loss)

train_8 = new_loss[0::2]
valid_8 = new_loss[1::2]

train_8 = np.array(train_8).astype(np.float32)
valid_8 = np.array(valid_8).astype(np.float32)


f = plt.figure(figsize=(20, 16))

print ("Plot saved in results folder")

plt.plot(valid_1, label=r"CIFAR-10, $MLE$", color='maroon', linestyle='--')
plt.plot(valid_2, label=r"CIFAR-100, $MLE$", color='darkblue', linestyle='--')
plt.plot(valid_3, label=r"MNIST, $MLE$", color='m', linestyle='--')
plt.plot(valid_4, label=r"STL10, $MLE$", color='green', linestyle='--')
plt.plot(valid_5, label=r"CIFAR-10, prior: $U(a, b)$", color='maroon')
plt.plot(valid_6, label=r"CIFAR-100, prior: $U(a, b)$", color='darkblue')
plt.plot(valid_7, label=r"MNIST, prior: $U(a, b)$", color='m')
plt.plot(valid_8, label=r"STL10, prior: $U(a, b)$", color='green')




plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
x_ticks = range(len(valid_1))
y_ticks = range(0,3)
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

plt.legend(fontsize=16)

plt.savefig("results_lenet_loss.png")

