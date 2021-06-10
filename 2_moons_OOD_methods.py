#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 07:17:13 2021

@author: root
"""
import sklearn
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
from utils_two_moons import evaluate_model, brier_score, expectation_calibration_error
from utils_two_moons import NeuralNet, MCDropout, EnsembleNeuralNet
from utils_two_moons import mixup_log_loss
from training_loops import train_model_dropout
from utils_two_moons import MyData
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

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
X_test,Y_test = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=.1, random_state=None)
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
test_dataset = MyData(data=X_test,labels=Y_test)
scale_dataset = MyData(X_scale, Y)
rot_dataset = MyData(X_rot, Y)
noise_dataset = MyData(X_noise, Y_noise)

trainLoader = DataLoader(train_dataset, batch_size=batch_sample)  
testLoader = DataLoader(test_dataset, batch_size=batch_sample)  
scaleLoader = DataLoader(scale_dataset, batch_size=batch_sample)    
rotLoader = DataLoader(rot_dataset, batch_size=batch_sample)    
noiseLoader = DataLoader(noise_dataset, batch_size=batch_sample)   

################# 2.TRAINING ################# 
# Simple Neural Network
base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 500
_, training_loss = train_model_dropout(base_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2)

# Neural Network with MC Dropout
vi_nn = MCDropout(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(vi_nn.parameters(), lr=0.01)
MC_sample=50
crit = nn.CrossEntropyLoss()
n_epochs = 500
_, training_loss = train_model_dropout(vi_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2)

# project the 64-dimensional data to a lower dimension
def estimate_input_density(data):
# project the 64-dimensional data to a lower dimension
    pca = PCA(n_components=2, whiten=False)
    data = pca.fit_transform(data)
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    return kde, pca

kde, pca = estimate_input_density(X)

## Train an ensemble of NN
def train_ensemble(N, n_epochs, trainLoader):
    ensembles = []
    for i in range(N):
        base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
        optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
        MC_sample=1
        crit = nn.CrossEntropyLoss()
        _, training_loss = train_model_dropout(base_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2)
        ensembles.append(base_nn)
    return ensembles

ensemble = train_ensemble(5, 500, trainLoader)

ensemble_nn = EnsembleNeuralNet(ensemble)

## Train with mixup
# Simple Neural Network
mu_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(mu_nn.parameters(), lr=0.01)
MC_sample=1
crit = mixup_log_loss
n_epochs = 500
_, training_loss = train_model_dropout(mu_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2, mixup=True)

## Train Fast Gradient Sign Method
# Simple Neural Network
fgsm_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(fgsm_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 500
_, training_loss = train_model_dropout(fgsm_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2, mixup=False, fgsm=True)
#plt.plot(training_loss)

################# 3.EVALUATE ################# 

### 3.0 Accuracy
print("--------------")
base_test_acc = evaluate_model(base_nn, testLoader,MC_sample=1,no_classes=2)
base_scale_acc = evaluate_model(base_nn, scaleLoader,1,2)
base_rot_acc = evaluate_model(base_nn, rotLoader,1,2)
base_noise_acc = evaluate_model(base_nn, noiseLoader,1,2)
print("Accuracy Softmax\n test_acc: {}\n scale_acc: {}\n rot_acc: {}\n noise_acc: {}".format(base_test_acc[0], base_scale_acc[0], base_rot_acc[0], base_noise_acc[0]))

print("--------------")
vi_test_acc = evaluate_model(vi_nn, testLoader, MC_sample=50,no_classes=2)
vi_scale_acc = evaluate_model(vi_nn, scaleLoader,50, 2)
vi_rot_acc = evaluate_model(vi_nn, rotLoader, 50, 2)
vi_noise_acc = evaluate_model(vi_nn, noiseLoader, 50,2)
print("Accuracy Dropout\n vi_test_acc: {}\n vi_scale_acc: {}\n vi_rot_acc: {}\n vi_noise_acc: {}".format(vi_test_acc[0], vi_scale_acc[0], vi_rot_acc[0], vi_noise_acc[0]))

print("--------------")
en_test_acc = evaluate_model(ensemble_nn, testLoader, MC_sample=1,no_classes=2)
en_scale_acc = evaluate_model(ensemble_nn, scaleLoader,1, 2)
en_rot_acc = evaluate_model(ensemble_nn, rotLoader, 1, 2)
en_noise_acc = evaluate_model(ensemble_nn, noiseLoader, 1,2)
print("Accuracy Ensemble\n en_test_acc: {}\n en_scale_acc: {}\n en_rot_acc: {}\n en_noise_acc: {}".format(en_test_acc[0], en_scale_acc[0], en_rot_acc[0], en_noise_acc[0]))

print("--------------")
mu_test_acc = evaluate_model(mu_nn, testLoader, MC_sample=1,no_classes=2)
mu_scale_acc = evaluate_model(mu_nn, scaleLoader,1, 2)
mu_rot_acc = evaluate_model(mu_nn, rotLoader, 1, 2)
mu_noise_acc = evaluate_model(mu_nn, noiseLoader, 1,2)
print("Accuracy Mixup\n mu_test_acc: {}\n mu_scale_acc: {}\n mu_rot_acc: {}\n mu_noise_acc: {}".format(mu_test_acc[0], mu_scale_acc[0], mu_rot_acc[0], mu_noise_acc[0]))

print("--------------")
ad_test_acc = evaluate_model(fgsm_nn, testLoader, MC_sample=1,no_classes=2)
ad_scale_acc = evaluate_model(fgsm_nn, scaleLoader,1, 2)
ad_rot_acc = evaluate_model(fgsm_nn, rotLoader, 1, 2)
ad_noise_acc = evaluate_model(fgsm_nn, noiseLoader, 1,2)
print("Accuracy FGSM\n ad_test_acc: {}\n ad_scale_acc: {}\n ad_rot_acc: {}\n ad_noise_acc: {}".format(ad_test_acc[0], ad_scale_acc[0], ad_rot_acc[0], ad_noise_acc[0]))

acc_losses = {(0,0):base_scale_acc, (0,1):base_rot_acc, (0,2):base_noise_acc, 
              (1,0):vi_scale_acc, (1,1):vi_rot_acc, (1,2):vi_noise_acc,
              (2,0):en_scale_acc, (2,1):en_rot_acc, (2,2):en_noise_acc,
              (3,0):mu_scale_acc, (3,1):mu_rot_acc, (3,2):mu_noise_acc,
              (4,0):ad_scale_acc, (4,1):ad_rot_acc, (4,2):ad_noise_acc              }


### Uncertainty
loss_functions = ["Softmax", "Dropout", "Ensemble", "Mixup", "FGSM"]
labels = ["Test", "Scale", "Rotation", "Noise"]
width = 0.25

### 3.1 Brier
base_bs_scale = brier_score(base_nn, scaleLoader, 1, 2)
base_bs_train = brier_score(base_nn, testLoader, 1, 2)
base_bs_rot = brier_score(base_nn, rotLoader, 1, 2)
base_bs_noise = brier_score(base_nn, noiseLoader, 1, 2)

vi_bs_scale = brier_score(vi_nn, scaleLoader, 50, 2)
vi_bs_train = brier_score(vi_nn, testLoader, 50, 2)
vi_bs_rot = brier_score(vi_nn, rotLoader, 50, 2)
vi_bs_noise = brier_score(vi_nn, noiseLoader, 50, 2)

en_bs_scale = brier_score(ensemble_nn, scaleLoader, 1, 2)
en_bs_train = brier_score(ensemble_nn, testLoader, 1, 2)
en_bs_rot = brier_score(ensemble_nn, rotLoader, 1, 2)
en_bs_noise = brier_score(ensemble_nn, noiseLoader, 1, 2)

mu_bs_scale = brier_score(mu_nn, scaleLoader, 1, 2)
mu_bs_train = brier_score(mu_nn, testLoader, 1, 2)
mu_bs_rot = brier_score(mu_nn, rotLoader, 1, 2)
mu_bs_noise = brier_score(mu_nn, noiseLoader, 1, 2)

ad_bs_scale = brier_score(fgsm_nn, scaleLoader, 1, 2)
ad_bs_train = brier_score(fgsm_nn, testLoader, 1, 2)
ad_bs_rot = brier_score(fgsm_nn, rotLoader, 1, 2)
ad_bs_noise = brier_score(fgsm_nn, noiseLoader, 1, 2)

# Aggregate all the scores
base_bs = [base_bs_train, base_bs_scale, base_bs_rot, base_bs_noise]
vi_bs = [vi_bs_train, vi_bs_scale, vi_bs_rot, vi_bs_noise]
en_bs = [en_bs_train, en_bs_scale, en_bs_rot, en_bs_noise]
mu_bs = [mu_bs_train, mu_bs_scale, mu_bs_rot, mu_bs_noise]
ad_bs = [ad_bs_train, ad_bs_scale, ad_bs_rot, ad_bs_noise]

ind = np.arange(0,8,2) 
bar1 = plt.bar(ind, base_bs, width)
bar2 = plt.bar(ind+width, vi_bs, width)
bar3 = plt.bar(ind+2*width, en_bs, width)
bar4 = plt.bar(ind+3*width, mu_bs, width)
bar5 = plt.bar(ind+4*width, mu_bs, width)
plt.xlabel("Distorsion")
plt.ylabel('Brier score')
plt.xticks(ind+width,labels)
plt.legend((bar1, bar2, bar3, bar4, bar5), loss_functions, loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("Brier", dpi=300)

#### 3.2 ECE

base_ece_scale = expectation_calibration_error(base_nn, scaleLoader, 1, 10, 2)
base_ece_test = expectation_calibration_error(base_nn, testLoader, 1, 10, 2)
base_ece_rot = expectation_calibration_error(base_nn, rotLoader, 1, 10, 2)
base_ece_noise = expectation_calibration_error(base_nn, noiseLoader, 1, 10, 2)

vi_ece_scale = expectation_calibration_error(vi_nn, scaleLoader, 50, 10, 2)
vi_ece_test = expectation_calibration_error(vi_nn, testLoader, 50, 10, 2)
vi_ece_rot = expectation_calibration_error(vi_nn, rotLoader, 50, 10, 2)
vi_ece_noise = expectation_calibration_error(vi_nn, noiseLoader, 50, 10, 2)

en_ece_scale = expectation_calibration_error(ensemble_nn, scaleLoader, 1, 10, 2)
en_ece_test = expectation_calibration_error(ensemble_nn, testLoader, 1, 10, 2)
en_ece_rot = expectation_calibration_error(ensemble_nn, rotLoader, 1, 10, 2)
en_ece_noise = expectation_calibration_error(ensemble_nn, noiseLoader, 1, 10, 2)

mu_ece_scale = expectation_calibration_error(mu_nn, scaleLoader, 1, 10, 2)
mu_ece_test = expectation_calibration_error(mu_nn, testLoader, 1, 10, 2)
mu_ece_rot = expectation_calibration_error(mu_nn, rotLoader, 1, 10, 2)
mu_ece_noise = expectation_calibration_error(mu_nn, noiseLoader, 1, 10, 2)

ad_ece_scale = expectation_calibration_error(fgsm_nn, scaleLoader, 1, 10, 2)
ad_ece_test = expectation_calibration_error(fgsm_nn, testLoader, 1, 10, 2)
ad_ece_rot = expectation_calibration_error(fgsm_nn, rotLoader, 1, 10, 2)
ad_ece_noise = expectation_calibration_error(fgsm_nn, noiseLoader, 1, 10, 2)


base_ece = [base_ece_test, base_ece_scale, base_ece_rot, base_ece_noise]
vi_ece = [vi_ece_test, vi_ece_scale, vi_ece_rot, vi_ece_noise]
en_ece = [en_ece_test, en_ece_scale, en_ece_rot, en_ece_noise]
mu_ece = [mu_ece_test, mu_ece_scale, mu_ece_rot, mu_ece_noise]
ad_ece = [ad_ece_test, ad_ece_scale, ad_ece_rot, ad_ece_noise]

ind = np.arange(0,8,2) 
bar1 = plt.bar(ind, base_ece, width)
bar2 = plt.bar(ind+width, vi_ece, width)
bar3 = plt.bar(ind+2*width, en_ece, width)
bar4 = plt.bar(ind+3*width, en_ece, width)
bar5 = plt.bar(ind+4*width, en_ece, width)
plt.xlabel("Distorsion")
plt.ylabel('ECE')
plt.xticks(ind+width,labels)
plt.legend((bar1, bar2, bar3, bar4, bar5), loss_functions, loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("ECE", dpi=300)

## AUC
# TODO: Develops the AUC for all models and datasets.
# Softmax
Y_test_pred = torch.sigmoid(base_nn(torch.tensor(X_test)))[:,1].detach().numpy()
Y_scale_pred = torch.sigmoid(base_nn(torch.tensor(X_scale)))[:,1].detach().numpy()
Y_rot_pred = torch.sigmoid(base_nn(torch.tensor(X_rot)))[:,1].detach().numpy()
Y_noise_pred = torch.sigmoid(base_nn(torch.tensor(X_noise)))[:,1].detach().numpy()

base_auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
base_auc_scale = sklearn.metrics.roc_auc_score(Y_test, Y_scale_pred)
base_auc_rot = sklearn.metrics.roc_auc_score(Y_test, Y_rot_pred)
base_auc_noise = sklearn.metrics.roc_auc_score(Y_test, Y_noise_pred)

base_auc = [base_auc_test, base_auc_scale, base_auc_rot, base_auc_noise]

# Dropout
Y_test_pred = torch.cat([torch.sigmoid(vi_nn(torch.tensor(X_test))) for i in range(50)],1).mean(1).detach().numpy()
Y_scale_pred = torch.cat([torch.sigmoid(vi_nn(torch.tensor(X_scale))) for i in range(50)],1).mean(1).detach().numpy()
Y_rot_pred = torch.cat([torch.sigmoid(vi_nn(torch.tensor(X_rot))) for i in range(50)],1).mean(1).detach().numpy()
Y_noise_pred = torch.cat([torch.sigmoid(vi_nn(torch.tensor(X_noise))) for i in range(50)],1).mean(1).detach().numpy()

vi_auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
vi_auc_scale = sklearn.metrics.roc_auc_score(Y_test, Y_scale_pred)
vi_auc_rot = sklearn.metrics.roc_auc_score(Y_test, Y_rot_pred)
vi_auc_noise = sklearn.metrics.roc_auc_score(Y_test, Y_noise_pred)

# Ensembles
Y_test_pred = torch.sigmoid(ensemble_nn(torch.tensor(X_test)))[:,1].detach().numpy()
Y_scale_pred = torch.sigmoid(ensemble_nn(torch.tensor(X_scale)))[:,1].detach().numpy()
Y_rot_pred = torch.sigmoid(ensemble_nn(torch.tensor(X_rot)))[:,1].detach().numpy()
Y_noise_pred = torch.sigmoid(ensemble_nn(torch.tensor(X_noise)))[:,1].detach().numpy()

en_auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
en_auc_scale = sklearn.metrics.roc_auc_score(Y_test, Y_scale_pred)
en_auc_rot = sklearn.metrics.roc_auc_score(Y_test, Y_rot_pred)
en_auc_noise = sklearn.metrics.roc_auc_score(Y_test, Y_noise_pred)

# Mixup
Y_test_pred = torch.sigmoid(mu_nn(torch.tensor(X_test)))[:,1].detach().numpy()
Y_scale_pred = torch.sigmoid(mu_nn(torch.tensor(X_scale)))[:,1].detach().numpy()
Y_rot_pred = torch.sigmoid(mu_nn(torch.tensor(X_rot)))[:,1].detach().numpy()
Y_noise_pred = torch.sigmoid(mu_nn(torch.tensor(X_noise)))[:,1].detach().numpy()

mu_auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
mu_auc_scale = sklearn.metrics.roc_auc_score(Y_test, Y_scale_pred)
mu_auc_rot = sklearn.metrics.roc_auc_score(Y_test, Y_rot_pred)
mu_auc_noise = sklearn.metrics.roc_auc_score(Y_test, Y_noise_pred)

# FGSM
Y_test_pred = torch.sigmoid(fgsm_nn(torch.tensor(X_test)))[:,1].detach().numpy()
Y_scale_pred = torch.sigmoid(fgsm_nn(torch.tensor(X_scale)))[:,1].detach().numpy()
Y_rot_pred = torch.sigmoid(fgsm_nn(torch.tensor(X_rot)))[:,1].detach().numpy()
Y_noise_pred = torch.sigmoid(fgsm_nn(torch.tensor(X_noise)))[:,1].detach().numpy()

ad_auc_test = sklearn.metrics.roc_auc_score(Y_test, Y_test_pred)
ad_auc_scale = sklearn.metrics.roc_auc_score(Y_test, Y_scale_pred)
ad_auc_rot = sklearn.metrics.roc_auc_score(Y_test, Y_rot_pred)
ad_auc_noise = sklearn.metrics.roc_auc_score(Y_test, Y_noise_pred)

# Aggreagating

vi_auc = [vi_auc_test, vi_auc_scale, vi_auc_rot, vi_auc_noise]
en_auc = [en_auc_test, en_auc_scale, en_auc_rot, en_auc_noise]
mu_auc = [mu_auc_test, mu_auc_scale, mu_auc_rot, mu_auc_noise]
ad_auc = [ad_auc_test, ad_auc_scale, ad_auc_rot, ad_auc_noise]

ind = np.arange(0,8,2) 
bar1 = plt.bar(ind, base_auc, width)
bar2 = plt.bar(ind+width, vi_auc, width)
bar3 = plt.bar(ind+2*width, en_auc, width)
bar4 = plt.bar(ind+3*width, en_auc, width)
bar5 = plt.bar(ind+4*width, en_auc, width)
plt.xlabel("Distorsion")
plt.ylabel('AUC')
plt.xticks(ind+width,labels)
plt.legend((bar1, bar2, bar3, bar4, bar5), loss_functions, loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("AUC", dpi=300)


################# 4.DRAW DECISION BOUNDARIES ################# 
def negatify(X):
    X = np.copy(X)
    neg = X < 0.5
    X[neg] =X[neg]-1 
    return X

# Create a mesh
h = .02  # step size in the mesh
x_min = np.concatenate([X[:, 0], X_rot[:, 0], X_scale[:, 0], X_noise[:, 0]]).min()
x_max = np.concatenate([X[:, 0], X_rot[:, 0], X_scale[:, 0], X_noise[:, 0]]).max()
y_min = np.concatenate([X[:, 1], X_rot[:, 1], X_scale[:, 1], X_noise[:, 1]]).min()
y_max = np.concatenate([X[:, 1], X_rot[:, 1], X_scale[:, 1], X_noise[:, 1]]).max()

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point of the mesh
base_Z = torch.sigmoid(base_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
# Here we create a list that we concatenate and we average the result
vi_Z = torch.cat([torch.sigmoid(vi_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1:]) for i in range(50)],1).mean(1)
en_Z = torch.sigmoid(ensemble_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
mu_Z = torch.sigmoid(mu_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
ad_Z = torch.sigmoid(fgsm_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])

base_Z = base_Z.reshape(xx.shape).detach().numpy()
base_Z_ = negatify(base_Z)
vi_Z = vi_Z.reshape(xx.shape).detach().numpy()
vi_Z_ = negatify(vi_Z)
en_Z = en_Z.reshape(xx.shape).detach().numpy()
en_Z_ = negatify(en_Z)
mu_Z = mu_Z.reshape(xx.shape).detach().numpy()
mu_Z_ = negatify(mu_Z)
ad_Z = ad_Z.reshape(xx.shape).detach().numpy()
ad_Z_ = negatify(ad_Z)



p_x = kde.score_samples(pca.transform(np.c_[xx.ravel(), yy.ravel()]))
p_x = p_x.reshape(xx.shape)
p_x_e = np.power(np.exp(1), p_x)
p_x_2 = np.power(2, p_x)
p_x_1_5 = np.power(1.5, p_x)


# Put the result into a color plot
cm = plt.cm.RdBu
plt.rcParams.update({'font.size': 14})
# Draw it!
fig, ax = plt.subplots(3,6, figsize=(32,16))
ax[0,0].set_title("Softmax")
ax[0,1].set_title("FGSM")
ax[0,2].set_title("Dropout")
ax[0,3].set_title("Mixup")
ax[0,4].set_title("Ensemble")
ax[0,5].set_title("PDE")
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
    for j in range(0,6):        
        if j==0:
            Z = base_Z
        elif j==1:
            Z = ad_Z 
        elif j==2:
            Z = vi_Z
        elif j==3:
            Z = mu_Z 
        elif j==4:
            Z = en_Z
        else:
            Z = base_Z_ * p_x_e
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)
        fig.colorbar(im, ax=ax[i,j])
        ax[i,j].scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
        ax[i,j].scatter(X_test[:, 0], X_test[:, 1], marker='+',c=Y, cmap=cm_bright)
#        ax[i,j].text(xx.max() - .3, yy.max() - .3, ('%.2f' % acc_losses[j,i][0]).lstrip('0'),
#                        size=15, horizontalalignment='right')
plt.savefig("New_OOD_score_5", dpi=300)





