#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 07:36:39 2021

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# load the data
cm_bright = ListedColormap(['#FF0000', '#0000FF'])      
batch_sample = 100        
X,Y = datasets.make_moons(n_samples=batch_sample, shuffle=True, noise=.1, random_state=None)
plt.scatter(X[:, 0], X[:, 1], c=Y)


# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=2, whiten=False)
data = pca.fit_transform(X)

# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

# sample 44 new points from the data and check that the learned distribution
# is accurate
new_data = kde.sample(100, random_state=0)
new_data = pca.inverse_transform(new_data)
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Create a mesh
h = .02  # step size in the mesh
x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point of the mesh
p_x = kde.score_samples(pca.transform(np.c_[xx.ravel(), yy.ravel()]))
#p_x = kde.score_samples(np.c_[xx.ravel(), yy.ravel()])
p_x = p_x.reshape(xx.shape)
p_x_e = np.power(np.exp(1), p_x)
p_x_2 = np.power(2, p_x)
p_x_1_5 = np.power(1.5, p_x)
p_x_1_1 = np.power(1.1, p_x)
p_x_1_01 = np.power(1.01, p_x)

# Put the result into a color plot

cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_e , cmap=cm, alpha=.8)
plt.colorbar()

def negatify(X):
    neg = X < 0.5
    X[neg] =X[neg]-1 
    return X

cm = plt.cm.RdBu
plt.scatter(new_data[:, 0], new_data[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x ,cmap=cm, alpha=.5)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_e ,cmap=cm, alpha=.5)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_2 ,cmap=cm, alpha=.5)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_1_5 ,cmap=cm, alpha=.5)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_1_1 ,cmap=cm, alpha=.5)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, p_x_1_01 ,cmap=cm, alpha=.5)
plt.colorbar()


## Now do it with base_Z
import torch
from utils_two_moons import NeuralNet, train_model, MyData
from torch.utils.data  import DataLoader

def empirical_log_loss(prediction, label):
#    import pdb; pdb.set_trace()
    log_probs_1 = torch.sigmoid(prediction)[:,1].log()
    log_probs_0 = (1-torch.sigmoid(prediction))[:,1].log()
#    import pdb; pdb.set_trace()
    log_loss = (label * log_probs_1 + (1-label)*(log_probs_0))
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return -log_loss.mean()

train_dataset = MyData(data=X,labels=Y)
trainLoader = DataLoader(train_dataset, batch_size=batch_sample)   


base_nn = NeuralNet(input_dim=2, hidden_dim=10, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
crit = empirical_log_loss
n_epochs = 500
_, training_loss = train_model(base_nn, base_nn, trainLoader, n_epochs, crit, optimizer)

base_Z = torch.sigmoid(base_nn(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))[:, 1])
base_Z = base_Z.reshape(xx.shape).detach().numpy()
base_Z = negatify(base_Z)

# Now Let's plot the uncertainties without p_x
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z , cmap=cm, alpha=.8)
plt.colorbar()

# Now Let's plot the uncertainties with p_x
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z_* p_x_e , cmap=cm, alpha=.8)
plt.colorbar()

# Now Let's plot the uncertainties with p_x_2
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z* p_x_2 , cmap=cm, alpha=.8)
plt.colorbar()

# Now Let's plot the uncertainties with p_x_1_5
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z* p_x_1_5 , cmap=cm, alpha=.8)
plt.colorbar()

# Now Let's plot the uncertainties with p_x_1_1
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z* p_x_1_1 , cmap=cm, alpha=.8)
plt.colorbar()

# Now Let's plot the uncertainties with p_x_1_01
cm = plt.cm.RdBu
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.contourf(xx, yy, base_Z* p_x_1_01 , cmap=cm, alpha=.8)
plt.colorbar()