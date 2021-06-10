#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:01:34 2021

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:01:58 2021

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
from training_loops import train_model_dropout, train_model_env
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

# We create the same dataset with more noise
X_noise,Y_noise = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=1, random_state=None)
plt.scatter(X_noise[:, 0], X_noise[:, 1], c=Y_noise)   

from scipy.stats import beta

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
a, b = 2, 10
rv = beta(a, b)
x= np.linspace(0,0.5,100)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
#x = np.linspace(beta.ppf(0.01, a, b),
#
#                beta.ppf(0.99, a, b), 100)
#ax.plot(x, beta.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='beta pdf')


# Create IID_training_set
total_environments = np.random.beta(2,10,100)
training_environments = total_environments[:70]
test_environments = total_environments[70:]

def get_environment_dataset(environments, batch_sample):
    environment_length = len(environments)
    data, labels = [], []
    for i in range(environment_length):
        local_data, local_labels = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=environments[i], random_state=None) 
        data.append(local_data)
        labels.append(local_labels)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

training_env_data, traning_env_labels = get_environment_dataset(training_environments, batch_sample)
iid_training_data, iid_training_labels = training_env_data[:,:70], traning_env_labels[:,:70]
iid_test_data, iid_test_labels = training_env_data[:,70:], traning_env_labels[:,70:]
ood_data, ood_labels =  get_environment_dataset(test_environments, batch_sample)

iid_training_data = iid_training_data.reshape([-1,2])
iid_training_labels = iid_training_labels.reshape([-1])
iid_test_data = iid_test_data.reshape([-1,2])
iid_test_labels = iid_test_labels.reshape([-1])

ood_data = ood_data.reshape([-1,2])
ood_labels = ood_labels.reshape([-1])


fig, ax = plt.subplots(1,3, figsize=(12,4))

ax[0].scatter(iid_training_data[:,0], iid_training_data[:,1], c=iid_training_labels)
ax[0].legend()
ax[0].set_title("IID Training Set")


ax[1].scatter(iid_test_data[:,0], iid_test_data[:,1], c=iid_test_labels)
ax[1].legend()
ax[1].set_title("IID Test Set")

ax[2].scatter(ood_data[:,0], ood_data[:,1], c=ood_labels)
ax[2].legend()
ax[2].set_title("OOD Test Set")

# Prepare data loaders
iid_train_dataset = MyData(data=iid_training_data,labels=iid_training_labels)
iid_test_dataset = MyData(data=iid_test_data,labels=iid_test_labels)
ood_ood_dataset = MyData(data=ood_data, labels=ood_labels)

batch_size = 256
iidTrainLoader = DataLoader(iid_train_dataset, batch_size=batch_size)  
iidTestLoader = DataLoader(iid_test_dataset, batch_size=batch_size)  
oodLoader = DataLoader(ood_ood_dataset, batch_size=batch_size)  


################# 2. TRAINING ################# 
# IRM
env_loader = DataLoader(MyData(training_env_data, traning_env_labels))
base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 200
_, train_loss, train_penalty = train_model_env(base_nn, None, MC_sample, env_loader, n_epochs, crit, optimizer, algo="IRM")
plt.plot(train_loss)
plt.plot(train_penalty)

# VREX
env_loader = DataLoader(MyData(training_env_data, traning_env_labels))
base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 200
_, train_loss, train_penalty = train_model_env(base_nn, None, MC_sample, env_loader, n_epochs, crit, optimizer, algo="VREX")
plt.plot(train_loss)
plt.plot(train_penalty)

# Simple Neural Network
base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 200
_, training_loss = train_model_dropout(base_nn, None, MC_sample, iidTrainLoader, n_epochs, crit, optimizer, no_classes=2)

# Neural Network with MC Dropout
vi_nn = MCDropout(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(vi_nn.parameters(), lr=0.01)
MC_sample=50
crit = nn.CrossEntropyLoss()
n_epochs = 200
_, training_loss = train_model_dropout(vi_nn, None, MC_sample, iidTrainLoader, n_epochs, crit, optimizer, no_classes=2)

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

ensemble = train_ensemble(5, 500, iidTrainLoader)

ensemble_nn = EnsembleNeuralNet(ensemble)

## Train with mixup
# Simple Neural Network
mu_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(mu_nn.parameters(), lr=0.01)
MC_sample=1
crit = mixup_log_loss
n_epochs = 200
_, training_loss = train_model_dropout(mu_nn, None, MC_sample, iidTrainLoader, n_epochs, crit, optimizer, no_classes=2, mixup=True)

## Train Fast Gradient Sign Method
# Simple Neural Network
fgsm_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(fgsm_nn.parameters(), lr=0.01)
MC_sample=1
crit = nn.CrossEntropyLoss()
n_epochs = 200
_, training_loss = train_model_dropout(fgsm_nn, None, MC_sample, iidTrainLoader, n_epochs, crit, optimizer, no_classes=2, mixup=False, fgsm=True)

################# 3. EVALUATION ################# 
from uncertainty import sample_lowest_entropy, sample_highest_density, sample_lowest_entropy_highest_density
retained = [50, 60, 70, 80, 90, 100]

def model_accuracy_over_low_entropy_data_retained(model, data, label, MC_sample, no_classes):
    """
    This function will retain the data with the lowest entropy
    at 6 different levels and put them in loaders.
    Furthermore, the accuracy at each level will be computed and returned 
    along with the associated loaders.
    The accuracies can be used to plot how the accuracy drops while we increase data
    the loader allow to have access to the data sampled with the low entropy
    criterion.         
    """
    loader50 = sample_lowest_entropy(0.5, model, data, label, MC_sample, no_classes)
    loader60 = sample_lowest_entropy(0.6, model, data, label, MC_sample, no_classes)
    loader70 = sample_lowest_entropy(0.7, model, data, label, MC_sample, no_classes)
    loader80 = sample_lowest_entropy(0.8, model, data, label, MC_sample, no_classes)
    loader90 = sample_lowest_entropy(0.9, model, data, label, MC_sample, no_classes)
    loader100 = sample_lowest_entropy(1., model, data, label, MC_sample, no_classes)

    acc_50 = evaluate_model(model, loader50, MC_sample, no_classes=2)
    acc_60 = evaluate_model(model, loader60, MC_sample, no_classes=2)
    acc_70 = evaluate_model(model, loader70, MC_sample, no_classes=2)
    acc_80 = evaluate_model(model, loader80, MC_sample, no_classes=2)
    acc_90 = evaluate_model(model, loader90, MC_sample, no_classes=2)
    acc_100 = evaluate_model(model, loader100, MC_sample, no_classes=2)
    
    acc = [acc_50, acc_60, acc_70, acc_80, acc_90, acc_100]
    loaders = [loader50, loader60, loader70, loader80, loader90, loader100]
    return acc, loaders    


def evaluate_model_accuracy(model, loaders, MC_sample, no_classes):
    """
    For a given model, compute the accuracy on all dataset to consider
    """
    iidTrainLoader, iidTestLoader, oodLoader = loaders
    iid_train_acc = evaluate_model(model, iidTrainLoader, MC_sample,no_classes)
    iid_test_acc = evaluate_model(model, iidTestLoader, MC_sample,no_classes)
    ood_acc = evaluate_model(model, oodLoader, MC_sample,no_classes)
    return iid_train_acc, iid_test_acc, ood_acc
    
    
### 3.0 Accuracy
loaders = [iidTrainLoader, iidTestLoader, oodLoader]   
 
print("--------------")
base_iid_train_acc, base_iid_test_acc, base_ood_acc = evaluate_model_accuracy(base_nn, loaders, MC_sample=1, no_classes=2)
print("Accuracy Softmax\n iid_train_acc: {}\n iid_test_acc: {}\n ood_acc: {}".format(base_iid_train_acc, base_iid_test_acc, base_ood_acc))

print("--------------")
vi_iid_train_acc, vi_iid_test_acc, vi_ood_acc = evaluate_model_accuracy(vi_nn, loaders, MC_sample=50, no_classes=2)
print("Accuracy Dropout\n iid_train_acc: {}\n iid_test_acc: {}\n ood_acc: {}".format(vi_iid_train_acc, vi_iid_test_acc, vi_ood_acc))

print("--------------")
en_iid_train_acc, en_iid_test_acc, en_ood_acc = evaluate_model_accuracy(ensemble_nn, loaders, MC_sample=1, no_classes=2)
print("Accuracy Ensembles\n iid_train_acc: {}\n iid_test_acc: {}\n ood_acc: {}".format(en_iid_train_acc, en_iid_test_acc, en_ood_acc))

print("--------------")
mu_iid_train_acc, mu_iid_test_acc, mu_ood_acc = evaluate_model_accuracy(mu_nn, loaders, MC_sample=1, no_classes=2)
print("Accuracy Mixup\n iid_train_acc: {}\n iid_test_acc: {}\n ood_acc: {}".format(mu_iid_train_acc, mu_iid_test_acc, mu_ood_acc))

print("--------------")
ad_iid_train_acc, ad_iid_test_acc, ad_ood_acc = evaluate_model_accuracy(fgsm_nn, loaders, MC_sample=1, no_classes=2)
print("Accuracy FGSM\n iid_train_acc: {}\n iid_test_acc: {}\n ood_acc: {}".format(ad_iid_train_acc, ad_iid_test_acc, ad_ood_acc))
