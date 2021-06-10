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
batch_sample = 1000        
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

# Train using the density
#base_density_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
#optimizer = torch.optim.Adam(base_density_nn.parameters(), lr=0.01)
#MC_sample=1
#crit = nn.CrossEntropyLoss()
#n_epochs = 500
#_, training_loss = train_model_dropout(base_density_nn, None, MC_sample, trainLoader, n_epochs, crit, optimizer, no_classes=2, kde=kde, pca=pca)


################# 3.EVALUATION BASED ON ACCURCAY ################# 
from uncertainty import sample_lowest_entropy, sample_highest_density, sample_lowest_entropy_highest_density
retained = [50, 60, 70, 80, 90, 100]

def model_accuracy_over_low_entropy_high_density_data_retained(model,kde, pca, data, label, MC_sample, no_classes):
    """
    This function will retain the data with the highest density
    at 6 different levels and put them in loaders.
    Furthermore, the accuracy at each level will be computed
    The accuracies can be used to plot how the accuracy drops while we increase data
    the loader allow to have access to the data sampled with the high density
    criterion.   
    """
    loader50 = sample_lowest_entropy_highest_density(.5, model, kde, pca, data, label, MC_sample, no_classes)
    loader60 = sample_lowest_entropy_highest_density(.6, model, kde, pca, data, label, MC_sample, no_classes)
    loader70 = sample_lowest_entropy_highest_density(.7, model, kde, pca, data, label, MC_sample, no_classes)
    loader80 = sample_lowest_entropy_highest_density(.8, model, kde, pca, data, label, MC_sample, no_classes)
    loader90 = sample_lowest_entropy_highest_density(.9, model, kde, pca, data, label, MC_sample, no_classes)
    loader100 = sample_lowest_entropy_highest_density(1., model, kde, pca, data, label, MC_sample, no_classes)
    
    acc_50 = evaluate_model(model, loader50, MC_sample, no_classes=2)
    acc_60 = evaluate_model(model, loader60, MC_sample, no_classes=2)
    acc_70 = evaluate_model(model, loader70, MC_sample, no_classes=2)
    acc_80 = evaluate_model(model, loader80, MC_sample, no_classes=2)
    acc_90 = evaluate_model(model, loader90, MC_sample, no_classes=2)
    acc_100 = evaluate_model(model, loader100, MC_sample, no_classes=2)
    
    acc = [acc_50, acc_60, acc_70, acc_80, acc_90, acc_100]
    loaders = [loader50, loader60, loader70, loader80, loader90, loader100]
    return acc, loaders

def model_accuracy_over_high_density_data_retained(model,kde, pca, data, label, MC_sample, no_classes):
    """
    This function will retain the data with the highest density
    at 6 different levels and put them in loaders.
    Furthermore, the accuracy at each level will be computed
    The accuracies can be used to plot how the accuracy drops while we increase data
    the loader allow to have access to the data sampled with the high density
    criterion.   
    """
    loader50 = sample_highest_density(0.5, kde, pca, data, label)
    loader60 = sample_highest_density(0.6, kde, pca, data, label)
    loader70 = sample_highest_density(0.7, kde, pca, data, label)
    loader80 = sample_highest_density(0.8, kde, pca, data, label)
    loader90 = sample_highest_density(0.9, kde, pca, data, label)
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

### Comparing sampling methods against each others
    
def aggregate_accuracy_perturbation_retained_data(model, kde, pca, datasets, labels, MC_sample, no_classes):
    X_test, X_scale, X_rot, X_noise = datasets
    Y_test, Y, Y_noise = labels
    test_ende_acc, test_ende_loaders = model_accuracy_over_low_entropy_high_density_data_retained(model,kde, pca, X_test, Y_test, MC_sample=1, no_classes=2)
    test_en_acc, test_en_loaders = model_accuracy_over_low_entropy_data_retained(model, X_test, Y_test, MC_sample=1, no_classes=2)
    test_de_acc, test_de_loaders = model_accuracy_over_high_density_data_retained(model,kde, pca, X_test, Y_test, MC_sample=1, no_classes=2)
        
    scale_ende_acc, scale_ende_loaders = model_accuracy_over_low_entropy_high_density_data_retained(model,kde, pca, X_scale, Y, MC_sample=1, no_classes=2)
    scale_en_acc, scale_en_loaders = model_accuracy_over_low_entropy_data_retained(model, X_scale, Y, MC_sample=1, no_classes=2)
    scale_de_acc, scale_de_loaders = model_accuracy_over_high_density_data_retained(model,kde, pca, X_scale, Y, MC_sample=1, no_classes=2)
        
    noise_ende_acc, noise_ende_loaders = model_accuracy_over_low_entropy_high_density_data_retained(model,kde, pca, X_noise, Y_noise, MC_sample=1, no_classes=2)
    noise_en_acc, noise_en_loaders = model_accuracy_over_low_entropy_data_retained(model, X_noise, Y_noise, MC_sample=1, no_classes=2)
    noise_de_acc, noise_de_loaders = model_accuracy_over_high_density_data_retained(model,kde, pca, X_noise, Y_noise, MC_sample=1, no_classes=2)
       
    rot_ende_acc, rot_ende_loaders = model_accuracy_over_low_entropy_high_density_data_retained(model,kde, pca, X_rot, Y, MC_sample=1, no_classes=2)
    rot_en_acc, rot_en_loaders = model_accuracy_over_low_entropy_data_retained(model, X_rot, Y, MC_sample=1, no_classes=2)
    rot_de_acc, rot_de_loaders = model_accuracy_over_high_density_data_retained(model,kde, pca, X_rot, Y, MC_sample=1, no_classes=2)
    
    aggregate_ende = np.concatenate([test_ende_acc, scale_ende_acc, noise_ende_acc, rot_ende_acc], 1)
    aggregate_en = np.concatenate([test_en_acc, scale_en_acc, noise_en_acc, rot_en_acc], 1)
    aggregate_de = np.concatenate([test_de_acc, scale_de_acc, noise_de_acc, rot_de_acc], 1)
    
    loaders_ende = [test_ende_loaders, scale_ende_loaders, noise_ende_loaders, rot_ende_loaders]
    loaders_en = [test_en_loaders, scale_en_loaders, noise_en_loaders, rot_en_loaders]
    loaders_de = [test_de_loaders, scale_de_loaders, noise_de_loaders, rot_de_loaders]
            
    return (aggregate_ende, aggregate_en, aggregate_de), (loaders_ende, loaders_en, loaders_de)
    
datasets = [X_test, X_scale, X_rot, X_noise]
labels = [Y_test, Y, Y_noise]

(base_ende, base_en, base_de), base_loaders = aggregate_accuracy_perturbation_retained_data(base_nn, kde, pca, datasets, labels, 1, 2)    
vi_ende, vi_en, vi_de = aggregate_accuracy_perturbation_retained_data(vi_nn, kde, pca, datasets, labels, 50, 2)    
en_ende, en_en, en_de = aggregate_accuracy_perturbation_retained_data(ensemble_nn, kde, pca, datasets, labels, 1, 2)    
mu_ende, mu_en, mu_de = aggregate_accuracy_perturbation_retained_data(mu_nn, kde, pca, datasets, labels, 1, 2)    
ad_ende, ad_en, ad_de = aggregate_accuracy_perturbation_retained_data(fgsm_nn, kde, pca, datasets, labels, 1, 2)    



fig, ax = plt.subplots(1,5, figsize=(22,4))

ax[0].set_ylabel("Aggregate over perturbations")
ax[0].plot(base_ende.mean(1), label="Entropy-Density")
ax[0].plot(base_en.mean(1), label="Entropy")
ax[0].plot(base_de.mean(1), label="Density")
ax[0].legend()
ax[0].set_title("Softmax")

ax[1].plot(vi_ende.mean(1), label="Entropy-Density")
ax[1].plot(vi_en.mean(1), label="Entropy")
ax[1].plot(vi_de.mean(1), label="Density")
ax[1].legend()
ax[1].set_title("Dropout")

ax[2].plot(en_ende.mean(1), label="Entropy-Density")
ax[2].plot(en_en.mean(1), label="Entropy")
ax[2].plot(en_de.mean(1), label="Density")
ax[2].legend()
ax[2].set_title("Ensemble")

ax[3].plot(mu_ende.mean(1), label="Entropy-Density")
ax[3].plot(mu_en.mean(1), label="Entropy")
ax[3].plot(mu_de.mean(1), label="Density")
ax[3].legend()
ax[3].set_title("Mixup")

ax[4].plot(ad_ende.mean(1), label="Entropy-Density")
ax[4].plot(ad_en.mean(1), label="Entropy")
ax[4].plot(ad_de.mean(1), label="Density")
ax[4].legend()
ax[4].set_title("FGSM")
plt.savefig("retained_aggregate_over_perturbation")

# Plot the aggregate accuracy with data retained
fig, ax = plt.subplots(1,4, figsize=(22,4))
ax[0].plot(base_en[0], label="Entropy")
ax[0].plot(base_de[0], label="Density")
#ax[0].plot(base_test_de2_acc, label="Density relaxed 2")
#ax[0].plot(base_test_de1_1_acc, label="Density relaxed 1.1")
ax[0].plot(base_ende[0], label="Entropy-Density")
ax[0].legend()
ax[0].set_title("Test data")

ax[1].plot(base_en[1], label="Entropy")
ax[1].plot(base_de[1], label="Density")
#ax[1].plot(base_scale_de2_acc, label="Density relaxed 2")
#ax[1].plot(base_scale_de1_1_acc, label="Density relaxed 1.1")
ax[1].plot(base_ende[1], label="Entropy-Density")
ax[1].legend()
ax[1].set_title("Scale data")

ax[2].plot(base_en[2], label="Entropy")
ax[2].plot(base_de[2], label="Density")
#ax[2].plot(base_noise_de2_acc, label="Density relaxed 2")
#ax[2].plot(base_noise_de1_1_acc, label="Density relaxed 1.1")
ax[2].plot(base_ende[2], label="Entropy-Density")
ax[2].legend()
ax[2].set_title("Noise data")

ax[3].plot(base_en[3], label="Entropy")
ax[3].plot(base_de[3], label="Density")
#ax[3].plot(base_rot_de2_acc, label="Density relaxed 2")
#ax[3].plot(base_rot_de1_1_acc, label="Density relaxed 1.1")
ax[3].plot(base_ende[3], label="Entropy-Density")
ax[3].legend()
ax[3].set_title("Rotation data")
plt.savefig("retained_lowestEntropy_highestDensity")


### Comparing methods agains each others
# Accuracies for data retained on the test set
base_test_acc, base_test_loaders = model_accuracy_over_low_entropy_data_retained(base_nn, X_test, Y_test, MC_sample=1, no_classes=2)
vi_test_acc, vi_test_loaders = model_accuracy_over_low_entropy_data_retained(vi_nn, X_test, Y_test, MC_sample=50, no_classes=2)
en_test_acc, en_test_loaders = model_accuracy_over_low_entropy_data_retained(ensemble_nn, X_test, Y_test, MC_sample=1, no_classes=2)
mu_test_acc, mu_test_loaders = model_accuracy_over_low_entropy_data_retained(mu_nn, X_test, Y_test, MC_sample=1, no_classes=2)
ad_test_acc, ad_test_loaders = model_accuracy_over_low_entropy_data_retained(fgsm_nn, X_test, Y_test, MC_sample=1, no_classes=2)
pde_test_acc, pde_test_loaders = model_accuracy_over_high_density_data_retained(base_nn,kde, pca, X_test, Y_test, MC_sample=1, no_classes=2)

# Accuracies for data retained on the scale perturbation set
base_scale_acc, base_scale_loaders = model_accuracy_over_low_entropy_data_retained(base_nn, X_scale, Y, MC_sample=1, no_classes=2)
vi_scale_acc, vi_scale_loaders = model_accuracy_over_low_entropy_data_retained(vi_nn, X_scale, Y, MC_sample=50, no_classes=2)
en_scale_acc, en_scale_loaders  = model_accuracy_over_low_entropy_data_retained(ensemble_nn, X_scale, Y, MC_sample=1, no_classes=2)
mu_scale_acc, mu_scale_loaders  = model_accuracy_over_low_entropy_data_retained(mu_nn, X_scale, Y, MC_sample=1, no_classes=2)
ad_scale_acc, ad_scale_loaders  = model_accuracy_over_low_entropy_data_retained(fgsm_nn, X_scale, Y, MC_sample=1, no_classes=2)
pde_scale_acc, pde_scale_loaders  = model_accuracy_over_high_density_data_retained(base_nn,kde, pca, X_scale, Y, MC_sample=1, no_classes=2)

# Accuracies for data retained on the scale rotation set
base_rot_acc, base_rot_loaders = model_accuracy_over_low_entropy_data_retained(base_nn, X_rot, Y, MC_sample=1, no_classes=2)
vi_rot_acc, vi_rot_loaders = model_accuracy_over_low_entropy_data_retained(vi_nn, X_rot, Y, MC_sample=50, no_classes=2)
en_rot_acc, en_rot_loaders  = model_accuracy_over_low_entropy_data_retained(ensemble_nn, X_rot, Y, MC_sample=1, no_classes=2)
mu_rot_acc, mu_rot_loaders  = model_accuracy_over_low_entropy_data_retained(mu_nn, X_rot, Y, MC_sample=1, no_classes=2)
ad_rot_acc, ad_rot_loaders  = model_accuracy_over_low_entropy_data_retained(fgsm_nn, X_rot, Y, MC_sample=1, no_classes=2)
pde_rot_acc, pde_rot_loaders  = model_accuracy_over_high_density_data_retained(base_nn,kde, pca, X_rot, Y, MC_sample=1, no_classes=2)

# Accuracies for data retained on the scale noise set
base_noise_acc, base_noise_loaders = model_accuracy_over_low_entropy_data_retained(base_nn, X_noise, Y_noise, MC_sample=1, no_classes=2)
vi_noise_acc, vi_noise_loaders = model_accuracy_over_low_entropy_data_retained(vi_nn, X_noise, Y_noise, MC_sample=50, no_classes=2)
en_noise_acc, en_noise_loaders = model_accuracy_over_low_entropy_data_retained(ensemble_nn, X_noise, Y_noise, MC_sample=1, no_classes=2)
mu_noise_acc, mu_noise_loaders = model_accuracy_over_low_entropy_data_retained(mu_nn, X_noise, Y_noise, MC_sample=1, no_classes=2)
ad_noise_acc, ad_noise_loaders = model_accuracy_over_low_entropy_data_retained(fgsm_nn, X_noise, Y_noise, MC_sample=1, no_classes=2)
pde_noise_acc, pde_noise_loaders = model_accuracy_over_high_density_data_retained(base_nn,kde, pca, X_noise, Y_noise, MC_sample=1, no_classes=2)


# Plot the aggregate accuracy with data retained
fig, ax = plt.subplots(1,4, figsize=(22,4))

ax[0].plot(retained, base_test_acc, label="Base")
ax[0].plot(retained, vi_test_acc, label="Dropout")
ax[0].plot(retained, en_test_acc, label="Ensemble")
ax[0].plot(retained, mu_test_acc, label="Mixup")
ax[0].plot(retained, ad_test_acc, label="FGSM")
ax[0].plot(retained, pde_test_acc, label="PDE")
ax[0].set_title("Test Set")

ax[1].plot(retained, base_scale_acc, label="Base")
ax[1].plot(retained, vi_scale_acc, label="Dropout")
ax[1].plot(retained, en_scale_acc, label="Ensemble")
ax[1].plot(retained, mu_scale_acc, label="Mixup")
ax[1].plot(retained, ad_scale_acc, label="FGSM")
ax[1].plot(retained, pde_scale_acc, label="PDE")
ax[1].set_title("Scale Perturbation")

ax[2].plot(retained, base_rot_acc, label="Base")
ax[2].plot(retained, vi_rot_acc, label="Dropout")
ax[2].plot(retained, en_rot_acc, label="Ensemble")
ax[2].plot(retained, mu_rot_acc, label="Mixup")
ax[2].plot(retained, ad_rot_acc, label="FGSM")
ax[2].plot(retained, pde_rot_acc, label="PDE")
ax[2].set_title("Rotation Perturbation")

ax[3].plot(retained, base_noise_acc, label="Base")
ax[3].plot(retained, vi_noise_acc, label="Dropout")
ax[3].plot(retained, en_noise_acc, label="Ensemble")
ax[3].plot(retained, mu_noise_acc, label="Mixup")
ax[3].plot(retained, ad_noise_acc, label="FGSM")
ax[3].plot(retained, pde_noise_acc, label="PDE")
ax[3].set_title("Noise Perturbation")
ax[3].legend(loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("retained_aggregate_accuracy", dpi=300)


################ 4. EVALUATION BASED ON AUC ################ 

def compute_auc_models(model, loaders, vi=False):
    loader50, loader60, loader70, loader80, loader90, loader100 = loaders
    
    if vi==True:
        Y_pred50 = torch.cat([torch.sigmoid(model(torch.tensor(loader50.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
        Y_pred60 = torch.cat([torch.sigmoid(model(torch.tensor(loader60.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
        Y_pred70 = torch.cat([torch.sigmoid(model(torch.tensor(loader70.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
        Y_pred80 = torch.cat([torch.sigmoid(model(torch.tensor(loader80.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
        Y_pred90 = torch.cat([torch.sigmoid(model(torch.tensor(loader90.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
        Y_pred100 = torch.cat([torch.sigmoid(model(torch.tensor(loader100.dataset.data)))[:,1:] for i in range(50)],1).mean(1).detach().numpy()
    else:
        Y_pred50 = torch.sigmoid(model(torch.tensor(loader50.dataset.data)))[:,1].detach().numpy()
        Y_pred60 = torch.sigmoid(model(torch.tensor(loader60.dataset.data)))[:,1].detach().numpy()
        Y_pred70 = torch.sigmoid(model(torch.tensor(loader70.dataset.data)))[:,1].detach().numpy()
        Y_pred80 = torch.sigmoid(model(torch.tensor(loader80.dataset.data)))[:,1].detach().numpy()
        Y_pred90 = torch.sigmoid(model(torch.tensor(loader90.dataset.data)))[:,1].detach().numpy()
        Y_pred100 = torch.sigmoid(model(torch.tensor(loader100.dataset.data)))[:,1].detach().numpy()
        
    auc50 = sklearn.metrics.roc_auc_score(loader50.dataset.labels, Y_pred50)
    auc60 = sklearn.metrics.roc_auc_score(loader60.dataset.labels, Y_pred60)
    auc70 = sklearn.metrics.roc_auc_score(loader70.dataset.labels, Y_pred70)
    auc80 = sklearn.metrics.roc_auc_score(loader80.dataset.labels, Y_pred80)
    auc90 = sklearn.metrics.roc_auc_score(loader90.dataset.labels, Y_pred90)
    auc100 = sklearn.metrics.roc_auc_score(loader100.dataset.labels, Y_pred100)    
    
    return [auc50, auc60, auc70, auc80, auc90, auc100]

# AUC for data retained on the test set
base_auc_test = compute_auc_models(base_nn, base_test_loaders, vi=False)
vi_auc_test = compute_auc_models(vi_nn, vi_test_loaders, vi=True)
en_auc_test = compute_auc_models(ensemble_nn, en_test_loaders, vi=False)
mu_auc_test = compute_auc_models(mu_nn, mu_test_loaders, vi=False)
ad_auc_test = compute_auc_models(fgsm_nn, ad_test_loaders, vi=False)
pde_auc_test = compute_auc_models(base_nn, pde_test_loaders, vi=False) 

# AUC for data retained on the scale perturbation set
base_auc_scale = compute_auc_models(base_nn, base_scale_loaders, vi=False)
vi_auc_scale = compute_auc_models(vi_nn, vi_scale_loaders, vi=True)
en_auc_scale = compute_auc_models(ensemble_nn, en_scale_loaders, vi=False)
mu_auc_scale = compute_auc_models(mu_nn, mu_scale_loaders, vi=False)
ad_auc_scale = compute_auc_models(fgsm_nn, ad_scale_loaders, vi=False)
pde_auc_scale = compute_auc_models(base_nn, pde_scale_loaders, vi=False) 

# AUC for data retained on the rotation perturbation set
base_auc_rot = compute_auc_models(base_nn, base_rot_loaders, vi=False)
vi_auc_rot = compute_auc_models(vi_nn, vi_rot_loaders, vi=True)
en_auc_rot = compute_auc_models(ensemble_nn, en_rot_loaders, vi=False)
mu_auc_rot = compute_auc_models(mu_nn, mu_rot_loaders, vi=False)
ad_auc_rot = compute_auc_models(fgsm_nn, ad_rot_loaders, vi=False)
pde_auc_rot = compute_auc_models(base_nn, pde_rot_loaders, vi=False) 

# AUC for data retained on the noise perturbation set
base_auc_noise = compute_auc_models(base_nn, base_noise_loaders, vi=False)
vi_auc_noise = compute_auc_models(vi_nn, vi_noise_loaders, vi=True)
en_auc_noise = compute_auc_models(ensemble_nn, en_noise_loaders, vi=False)
mu_auc_noise = compute_auc_models(mu_nn, mu_noise_loaders, vi=False)
ad_auc_noise = compute_auc_models(fgsm_nn, ad_noise_loaders, vi=False)
pde_auc_noise = compute_auc_models(base_nn, pde_noise_loaders, vi=False) 

# Plot the aggregate accuracy with data retained
fig, ax = plt.subplots(1,4, figsize=(22,4))

ax[0].plot(retained, base_auc_test, label="Base")
ax[0].plot(retained, vi_auc_test, label="Dropout")
ax[0].plot(retained, en_auc_test, label="Ensemble")
ax[0].plot(retained, mu_auc_test, label="Mixup")
ax[0].plot(retained, ad_auc_test, label="FGSM")
ax[0].plot(retained, pde_auc_test, label="PDE")
ax[0].set_title("Test Set")

ax[1].plot(retained, base_auc_scale, label="Base")
ax[1].plot(retained, vi_auc_scale, label="Dropout")
ax[1].plot(retained, en_auc_scale, label="Ensemble")
ax[1].plot(retained, mu_auc_scale, label="Mixup")
ax[1].plot(retained, ad_auc_scale, label="FGSM")
ax[1].plot(retained, pde_auc_scale, label="PDE")
ax[1].set_title("Scale Perturbation")

ax[2].plot(retained, base_auc_rot, label="Base")
ax[2].plot(retained, vi_auc_rot, label="Dropout")
ax[2].plot(retained, en_auc_rot, label="Ensemble")
ax[2].plot(retained, mu_auc_rot, label="Mixup")
ax[2].plot(retained, ad_auc_rot, label="FGSM")
ax[2].plot(retained, pde_auc_rot, label="PDE")
ax[2].set_title("Rotation Perturbation")

ax[3].plot(retained, base_auc_noise, label="Base")
ax[3].plot(retained, vi_auc_noise, label="Dropout")
ax[3].plot(retained, en_auc_noise, label="Ensemble")
ax[3].plot(retained, mu_auc_noise, label="Mixup")
ax[3].plot(retained, ad_auc_noise, label="FGSM")
ax[3].plot(retained, pde_auc_noise, label="PDE")
ax[3].set_title("Noise Perturbation")
ax[3].legend(loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("retained_aggregate_auc", dpi=300)



################# 5.DRAW DECISION BOUNDARIES ################# 
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

cm = plt.cm.RdBu
plt.rcParams.update({'font.size': 14})

##### 5.1 Plot on the test dataset
fig, ax = plt.subplots(6,6, figsize=(24,22))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Softmax")
ax[1,0].set_ylabel("Dropout")
ax[2,0].set_ylabel("Ensemble")
ax[3,0].set_ylabel("Mixup")
ax[4,0].set_ylabel("FGSM")
ax[5,0].set_ylabel("PDE")

for i in range(0,6):
    if i==0:
        loaders = base_test_loaders
        Z = base_Z
    elif i==1:
        loaders = vi_test_loaders
        Z = vi_Z
    elif i==2:
        loaders = en_test_loaders
        Z = en_Z
    elif i==3:
        loaders = mu_test_loaders
        Z = mu_Z
    elif i==4:
        loaders = ad_test_loaders
        Z = ad_Z
    else:
        loaders = pde_test_loaders
        Z = base_Z_ * p_x_e
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_test", dpi=300)


##### 5.2 Plot on the scale dataset
fig, ax = plt.subplots(6,6, figsize=(24,22))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Softmax")
ax[1,0].set_ylabel("Dropout")
ax[2,0].set_ylabel("Ensemble")
ax[3,0].set_ylabel("Mixup")
ax[4,0].set_ylabel("FGSM")
ax[5,0].set_ylabel("PDE")

for i in range(0,6):
    if i==0:
        loaders = base_scale_loaders
        Z = base_Z
    elif i==1:
        loaders = vi_scale_loaders
        Z = vi_Z
    elif i==2:
        loaders = en_scale_loaders
        Z = en_Z
    elif i==3:
        loaders = mu_scale_loaders
        Z = mu_Z
    elif i==4:
        loaders = ad_scale_loaders
        Z = ad_Z
    else:
        loaders = pde_scale_loaders
        Z = base_Z_ * p_x_e
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_scale[:, 0], X_scale[:, 1], c=Y, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_scale", dpi=300)


##### 5.3 Plot on the rotation dataset
fig, ax = plt.subplots(6,6, figsize=(24,22))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Softmax")
ax[1,0].set_ylabel("Dropout")
ax[2,0].set_ylabel("Ensemble")
ax[3,0].set_ylabel("Mixup")
ax[4,0].set_ylabel("FGSM")
ax[5,0].set_ylabel("PDE")

for i in range(0,6):
    if i==0:
        loaders = base_rot_loaders
        Z = base_Z
    elif i==1:
        loaders = vi_rot_loaders
        Z = vi_Z
    elif i==2:
        loaders = en_rot_loaders
        Z = en_Z
    elif i==3:
        loaders = mu_rot_loaders
        Z = mu_Z
    elif i==4:
        loaders = ad_rot_loaders
        Z = ad_Z
    else:
        loaders = pde_rot_loaders
        Z = base_Z_ * p_x_e
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_rot[:, 0], X_rot[:, 1], c=Y, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_rot", dpi=300)


##### 5.4 Plot on the noise dataset
fig, ax = plt.subplots(6,6, figsize=(24,22))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Softmax")
ax[1,0].set_ylabel("Dropout")
ax[2,0].set_ylabel("Ensemble")
ax[3,0].set_ylabel("Mixup")
ax[4,0].set_ylabel("FGSM")
ax[5,0].set_ylabel("PDE")

for i in range(0,6):
    if i==0:
        loaders = base_noise_loaders
        Z = base_Z
    elif i==1:
        loaders = vi_noise_loaders
        Z = vi_Z
    elif i==2:
        loaders = en_noise_loaders
        Z = en_Z
    elif i==3:
        loaders = mu_noise_loaders
        Z = mu_Z
    elif i==4:
        loaders = ad_noise_loaders
        Z = ad_Z
    else:
        loaders = pde_noise_loaders
        Z = base_Z_ * p_x_e
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_noise[:, 0], X_noise[:, 1], c=Y_noise, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_noise", dpi=300)



##### 5.5 Compare ENDE, DE, EN with base_nn 
# on test dataset
fig, ax = plt.subplots(3,6, figsize=(24,18))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Entropy-Density")
ax[1,0].set_ylabel("Entropy")
ax[2,0].set_ylabel("Density")


for i in range(0,3):
    if i==0:
        loaders = base_loaders[0][0]
        Z = base_Z
    elif i==1:
        loaders = base_loaders[1][0]
        Z = base_Z
    else:
        loaders = base_loaders[2][0]
        Z = base_Z_ * p_x_e   
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_noise[:, 0], X_noise[:, 1], c=Y_noise, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_test_ende_en_de", dpi=300)

# on scale dataset
fig, ax = plt.subplots(3,6, figsize=(24,18))

ax[0,0].set_title("50 % retained")
ax[0,1].set_title("60 % retained")
ax[0,2].set_title("70 % retained")
ax[0,3].set_title("80 % retained")
ax[0,4].set_title("90 % retained")
ax[0,5].set_title("100 % retained")
ax[0,0].set_ylabel("Entropy-Density")
ax[1,0].set_ylabel("Entropy")
ax[2,0].set_ylabel("Density")


for i in range(0,3):
    if i==0:
        loaders = base_loaders[0][1]
        Z = base_Z
    elif i==1:
        loaders = base_loaders[1][1]
        Z = base_Z
    else:
        loaders = base_loaders[2][1]
        Z = base_Z_ * p_x_e   
    for j in range(0,6):
        base_x, base_y = next(iter(loaders[j]))                
        im = ax[i,j].contourf(xx, yy, Z, cmap=cm, alpha=.8)        
        ax[i,j].scatter(base_x[:, 0], base_x[:, 1], c=base_y, cmap=cm_bright)
        ax[i,j].scatter(X_noise[:, 0], X_noise[:, 1], c=Y_noise, cmap=cm_bright, alpha=0.1)
plt.savefig("retained_scale_ende_en_de", dpi=300)

