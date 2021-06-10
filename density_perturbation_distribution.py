#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:49:20 2021

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

from statsmodels.distributions.empirical_distribution import ECDF

px_train = kde.score_samples(pca.transform(X))
px_train = np.power(np.exp(1), px_train)

px_test = kde.score_samples(pca.transform(X_test))
px_test = np.power(np.exp(1), px_test)

px_scale = kde.score_samples(pca.transform(X_scale))
px_scale = np.power(np.exp(1), px_scale)

px_rot = kde.score_samples(pca.transform(X_rot))
px_rot = np.power(np.exp(1), px_rot)

px_noise = kde.score_samples(pca.transform(X_noise))
px_noise = np.power(np.exp(1), px_noise)


ecdf_train = ECDF(px_train)
ecdf_test = ECDF(px_test)
ecdf_scale = ECDF(px_scale)
ecdf_rot = ECDF(px_rot)
ecdf_noise = ECDF(px_noise)

plt.plot(ecdf_train.x, ecdf_train.y, label='training')
plt.plot(ecdf_test.x, ecdf_test.y, label='test')
plt.plot(ecdf_scale.x, ecdf_scale.y, label='scale')
plt.plot(ecdf_rot.x, ecdf_rot.y, label='rotation')
plt.plot(ecdf_noise.x, ecdf_noise.y, label='noise')
plt.legend()
plt.title("Data Density Distribution")
plt.show()

