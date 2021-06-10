#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:20:20 2021

@author: root
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import trange
from utils_two_moons import evaluate_model, brier_score, expectation_calibration_error
from utils_two_moons import NeuralNet, train_model
from utils_two_moons import MyData

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

    
################# 1.CREATE THE DATASETS ################# 

cm_bright = ListedColormap(['#FF0000', '#0000FF'])      
batch_sample = 100        
X,Y = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=.1, random_state=None)
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Scale in x and y directions
aug_x = (1.5 - 0.5) * np.random.rand() + 0.5
aug_y = (2.5 - 1.5) * np.random.rand() + 1.5
aug = np.array([aug_x, aug_y])

X_scale = X * aug
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.scatter(X_scale[:, 0], X_scale[:, 1], marker='+',c=Y, cmap=cm_bright, alpha=0.4)

## rotation of 45 degrees
theta = (np.pi/180)* -35
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
X_rot = np.dot(X,rotation_matrix)

plt.scatter(X[:, 0], X[:, 1], c=Y,cmap=cm_bright)
plt.scatter(X_rot[:, 0], X_rot[:, 1], marker='+', c=Y, cmap=cm_bright, alpha=0.4)

# We create the same dataset with more noise
X_noise,Y_noise = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=.3, random_state=None)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.scatter(X_noise[:, 0], X_noise[:, 1], marker='+', c=Y_noise, cmap=cm_bright, alpha=0.4)   


train_dataset = MyData(data=X,labels=Y)
scale_dataset = MyData(X_scale, Y)
rot_dataset = MyData(X_rot, Y)
noise_dataset = MyData(X_noise, Y_noise)

trainLoader = DataLoader(train_dataset, batch_size=batch_sample)    
scaleLoader = DataLoader(scale_dataset, batch_size=batch_sample)    
rotLoader = DataLoader(rot_dataset, batch_size=batch_sample)    
noiseLoader = DataLoader(noise_dataset, batch_size=batch_sample)   

################# 2.LEARNING ################# 


# We start with the log_loss, it will serve as the base conditional density
# estimation for the other losses

# Here note that log(1-sigmoid(f(x))) = log(sigmoid(-f(x)))
def log_loss(prediction, label):
#    import pdb; pdb.set_trace()
    loss = nn.LogSigmoid()
#    import pdb; pdb.set_trace()
    log_loss = (label * loss(prediction[:,1]) + (1-label)*loss(-prediction[:,1]))
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return -log_loss.mean()

def empirical_log_loss(prediction, label):
#    import pdb; pdb.set_trace()
    log_probs_1 = torch.sigmoid(prediction)[:,1].log()
    log_probs_0 = (1-torch.sigmoid(prediction))[:,1].log()
#    import pdb; pdb.set_trace()
    log_loss = (label * log_probs_1 + (1-label)*(log_probs_0))
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return -log_loss.mean()


def total_variation_loss(prediction, label, cond_density):    
    probs_1 = torch.sigmoid(prediction)[:,1]
    probs_0 = (1-torch.sigmoid(prediction))[:,1]
    cond_density_1 = cond_density[:,1]
    cond_density_0 = (1-cond_density)[:,1]
    loss =  0.5 * (label * torch.abs(1-probs_1/cond_density_1) + (1-label)*torch.abs(1-probs_0/cond_density_0))
    return loss.mean()

def hellinger_loss(prediction, label, cond_density):
    probs_1 = torch.sigmoid(prediction)[:,1]
    probs_0 = (1-torch.sigmoid(prediction))[:,1]
    cond_density_1 = cond_density[:,1]
    cond_density_0 = (1-cond_density)[:,1]
    loss = 0.5 * (label * (1-torch.sqrt(probs_1/cond_density_1))**2 + (1-label)*(1-torch.sqrt(probs_0/cond_density_0))**2)
#    import pdb; pdb.set_trace()
    return loss.mean()

def alpha_loss(prediction, label, cond_density, alpha=1.5):
    probs_1 = torch.clamp(torch.sigmoid(prediction)[:,1], 1e-20, 1-1e-20)
    probs_0 = 1-probs_1
    cond_density_1 = cond_density[:,1]
    cond_density_0 = (1-cond_density)[:,1]
#    import pdb; pdb.set_trace()
    loss =  (1/ (alpha*(1-alpha))) *( label * (1-(cond_density_1**(alpha-1))*(probs_1**(1-alpha))) + (1-label) * (1-(cond_density_0**(alpha-1))*(probs_0**(1-alpha))) )   
    return loss.mean()

base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
crit = empirical_log_loss
n_epochs = 500
_, training_loss = train_model(base_nn, base_nn, trainLoader, n_epochs, crit, optimizer)

plt.plot(training_loss)

def train_with_customloss(crit, base_nn, n_epochs = 750):
    model = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _, training_loss = train_model(model, base_nn, trainLoader, n_epochs, crit, optimizer)
    return model, training_loss


tv_nn, tv_loss = train_with_customloss(total_variation_loss, base_nn)
he_nn, he_loss = train_with_customloss(hellinger_loss, base_nn)
a_nn, a_loss = train_with_customloss(alpha_loss, base_nn)



plt.plot(tv_loss[100:])
plt.plot(he_loss[100:])
plt.plot(a_loss[100:])

print("--------------")
train_acc = evaluate_model(base_nn, trainLoader)
scale_acc = evaluate_model(base_nn, scaleLoader)
rot_acc = evaluate_model(base_nn, rotLoader)
noise_acc = evaluate_model(base_nn, noiseLoader)
print("Log loss\n training_acc: {}\n scale_acc: {}\n rot_acc: {}\n noise_acc: {}".format(train_acc[0], scale_acc[0], rot_acc[0], noise_acc[0]))

print("--------------")
tv_train_acc = evaluate_model(tv_nn, trainLoader)
tv_scale_acc = evaluate_model(tv_nn, scaleLoader)
tv_rot_acc = evaluate_model(tv_nn, rotLoader)
tv_noise_acc = evaluate_model(tv_nn, noiseLoader)
print("Total Variation loss\n tv_training_acc: {}\n tv_scale_acc: {}\n tv_rot_acc: {}\n tv_noise_acc: {}".format(tv_train_acc[0], tv_scale_acc[0], tv_rot_acc[0], tv_noise_acc[0]))

print("--------------")
he_train_acc = evaluate_model(he_nn, trainLoader)
he_scale_acc = evaluate_model(he_nn, scaleLoader)
he_rot_acc = evaluate_model(he_nn, rotLoader)
he_noise_acc = evaluate_model(he_nn, noiseLoader)
print("Hellinger loss\n he_training_acc: {}\n he_scale_acc: {}\n he_rot_acc: {}\n he_noise_acc: {}".format(he_train_acc[0], he_scale_acc[0], he_rot_acc[0], he_noise_acc[0]))

print("--------------")
a_train_acc = evaluate_model(a_nn, trainLoader)
a_scale_acc = evaluate_model(a_nn, scaleLoader)
a_rot_acc = evaluate_model(a_nn, rotLoader)
a_noise_acc = evaluate_model(a_nn, noiseLoader)
print("Alpha loss\n training_acc: {}\n a_scale_acc: {}\n a_rot_acc: {}\n a_noise_acc: {}".format(a_train_acc[0], a_scale_acc[0], a_rot_acc[0], a_noise_acc[0]))

acc_losses = {(0,0):scale_acc, (0,1):rot_acc, (0,2):noise_acc, 
              (1,0):tv_scale_acc, (1,1):tv_rot_acc, (1,2):tv_noise_acc,
              (2,0):he_scale_acc, (2,1):he_rot_acc, (2,2):he_noise_acc,
              (3,0):a_scale_acc, (3,1):a_rot_acc, (3,2):a_noise_acc}
################# 3.EVALUATE ################# 
loss_functions = ["Log_loss", "TV_loss", "Hellinger_loss", "Alpha_loss"]
labels = ["Train", "Scale", "Rotation", "Noise"]
width = 0.25

### 3.1 Brier
base_bs_scale = brier_score(base_nn, scaleLoader, 1, 2)
base_bs_train = brier_score(base_nn, trainLoader, 1, 2)
base_bs_rot = brier_score(base_nn, rotLoader, 1, 2)
base_bs_noise = brier_score(base_nn, noiseLoader, 1, 2)

tv_bs_scale = brier_score(tv_nn, scaleLoader, 1, 2)
tv_bs_train = brier_score(tv_nn, trainLoader, 1, 2)
tv_bs_rot = brier_score(tv_nn, rotLoader, 1, 2)
tv_bs_noise = brier_score(tv_nn, noiseLoader, 1, 2)

he_bs_scale = brier_score(he_nn, scaleLoader, 1, 2)
he_bs_train = brier_score(he_nn, trainLoader, 1, 2)
he_bs_rot = brier_score(he_nn, rotLoader, 1, 2)
he_bs_noise = brier_score(he_nn, noiseLoader, 1, 2)

a_bs_scale = brier_score(a_nn, scaleLoader, 1, 2)
a_bs_train = brier_score(a_nn, trainLoader, 1, 2)
a_bs_rot = brier_score(a_nn, rotLoader, 1, 2)
a_bs_noise = brier_score(a_nn, noiseLoader, 1, 2)

base_bs = [base_bs_train, base_bs_scale, base_bs_rot, base_bs_noise]
tv_bs = [tv_bs_train, tv_bs_scale, tv_bs_rot, tv_bs_noise]
he_bs = [he_bs_train, he_bs_scale, he_bs_rot, he_bs_noise]
a_bs = [a_bs_train, a_bs_scale, a_bs_rot, a_bs_noise]

ind = np.arange(0,8,2) 
bar1 = plt.bar(ind, base_bs, width)
bar2 = plt.bar(ind+width, tv_bs, width)
bar3 = plt.bar(ind+(2*width), he_bs, width)
bar4 = plt.bar(ind+ (3*width), a_bs, width)
plt.xlabel("Distorsion")
plt.ylabel('Brier score')
plt.xticks(ind+width,labels)
plt.legend((bar1, bar2, bar3, bar4), loss_functions)


#### 3.2 ECE

base_ece_scale = expectation_calibration_error(base_nn, scaleLoader, 1, 10, 2)
base_ece_train = expectation_calibration_error(base_nn, trainLoader, 1, 10, 2)
base_ece_rot = expectation_calibration_error(base_nn, rotLoader, 1, 10, 2)
base_ece_noise = expectation_calibration_error(base_nn, noiseLoader, 1, 10, 2)

tv_ece_scale = expectation_calibration_error(tv_nn, scaleLoader, 1, 10, 2)
tv_ece_train = expectation_calibration_error(tv_nn, trainLoader, 1, 10, 2)
tv_ece_rot = expectation_calibration_error(tv_nn, rotLoader, 1, 10, 2)
tv_ece_noise = expectation_calibration_error(tv_nn, noiseLoader, 1, 10, 2)

he_ece_scale = expectation_calibration_error(he_nn, scaleLoader, 1, 10, 2)
he_ece_train = expectation_calibration_error(he_nn, trainLoader, 1, 10, 2)
he_ece_rot = expectation_calibration_error(he_nn, rotLoader, 1, 10, 2)
he_ece_noise = expectation_calibration_error(he_nn, noiseLoader, 1, 10, 2)

a_ece_scale = expectation_calibration_error(a_nn, scaleLoader, 1, 10, 2)
a_ece_train = expectation_calibration_error(a_nn, trainLoader, 1, 10, 2)
a_ece_rot = expectation_calibration_error(a_nn, rotLoader, 1, 10, 2)
a_ece_noise = expectation_calibration_error(a_nn, noiseLoader, 1, 10, 2)

base_ece = [base_ece_train, base_ece_scale, base_ece_rot, base_ece_noise]
tv_ece = [tv_ece_train, tv_ece_scale, tv_ece_rot, tv_ece_noise]
he_ece = [he_ece_train, he_ece_scale, he_ece_rot, he_ece_noise]
a_ece = [a_ece_train, a_ece_scale, a_ece_rot, a_ece_noise]


ind = np.arange(0,8,2) 
bar1 = plt.bar(ind, base_ece, width)
bar2 = plt.bar(ind+width, tv_ece, width)
bar3 = plt.bar(ind+(2*width), he_ece, width)
bar4 = plt.bar(ind+ (3*width), a_ece, width)
plt.xlabel("Distorsion")
plt.ylabel('ECE')
plt.xticks(ind+width,labels)
plt.legend((bar1, bar2, bar3, bar4), loss_functions)





################# 4.DRAW DECISION BOUNDARIES ################# 
# Create a mesh
h = .02  # step size in the mesh
x_min = np.concatenate([X[:, 0], X_rot[:, 0], X_scale[:, 0], X_noise[:, 0]]).min()
x_max = np.concatenate([X[:, 0], X_rot[:, 0], X_scale[:, 0], X_noise[:, 0]]).max()
y_min = np.concatenate([X[:, 1], X_rot[:, 1], X_scale[:, 1], X_noise[:, 1]]).min()
y_max = np.concatenate([X[:, 1], X_rot[:, 1], X_scale[:, 1], X_noise[:, 1]]).max()

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point of the mesh
base_Z = torch.sigmoid(base_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
tv_Z = torch.sigmoid(tv_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
he_Z = torch.sigmoid(he_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
a_Z = torch.sigmoid(a_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])

base_Z = base_Z.reshape(xx.shape).detach().numpy()
tv_Z = tv_Z.reshape(xx.shape).detach().numpy()
he_Z = he_Z.reshape(xx.shape).detach().numpy()
a_Z = a_Z.reshape(xx.shape).detach().numpy()


# Put the result into a color plot
cm = plt.cm.RdBu
plt.rcParams.update({'font.size': 14})
# Draw it!
fig, ax = plt.subplots(3,4, figsize=(28,16))
ax[0,0].set_title("Log loss")
ax[0,1].set_title("Total Variation loss")
ax[0,2].set_title("Hellinger loss")
ax[0,3].set_title("Alpha loss")
ax[0,0].set_ylabel("Scale corruption")
ax[1,0].set_ylabel("Rotation corruption")
ax[2,0].set_ylabel("Noise corruption")
for i in range(0,3):
    if i==0:
        X_test = X_scale
    elif i==1:
        X_test = X_rot
    else:
        X_test = X_noise
    for j in range(0,4):        
        if j==0:
            Z = base_Z
        elif j==1:
            Z = tv_Z
        elif j==2:
            Z = he_Z
        else:
            Z = a_Z
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)
        fig.colorbar(im, ax=ax[i,j])
        ax[i,j].scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
        ax[i,j].scatter(X_test[:, 0], X_test[:, 1], marker='+',c=Y, cmap=cm_bright)
        ax[i,j].text(xx.max() - .3, yy.max() - .3, ('%.2f' % acc_losses[j,i][0]).lstrip('0'),
                        size=15, horizontalalignment='right')
plt.savefig("All_losses_score", dpi=300)




