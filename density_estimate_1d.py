#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:18:29 2021

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
import torch.nn as nn
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import trange


n_sample = 1000
mixture = np.random.binomial(n_sample, 0.99)
g1 = np.random.normal(2,1, mixture)
g2 = np.random.normal(7,5, n_sample-mixture)

np.random.normal()
g = np.concatenate([g1,g2])

ax = sns.distplot(g,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class MyData(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        """"Initialization, here display serves to show an example
        if false, it means that we intend to feed the data to a model"""
        self.data = data           

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]            
        # Load data and get label              
        return X

# @title Gaussian Layer and BayesianLinear Flipout class
class GaussianLayerLRT(nn.Module):
    def __init__(self, shape, init):
        super(GaussianLayerLRT, self).__init__()               
        if init == "prior":   
            # Here we use the [[]] to have the same dimension for the prior and posterior
          self.mu = nn.Parameter(torch.zeros(shape, ))
          self.log_std = nn.Parameter(torch.ones(shape)*np.log(10))
        elif init == "standard":
          self.mu = nn.Parameter(torch.zeros(shape, ))
          self.log_std = nn.Parameter(torch.ones(shape))
        else:
          self.mu = nn.Parameter(torch.zeros(shape).uniform_(-0.6, 0.6))
          self.log_std = nn.Parameter(torch.zeros(shape).uniform_(-6,0))

    def forward(self):
        return (self.mu, self.log_std.exp()) 
    
    def entropy(self):
        distribution = Normal(loc=self.mu, scale=self.log_std.exp())
        return distribution.entropy().mean() 
    
    def KL_weight(self, mu_2, log_std_2):
        mu_1 = self.mu
        log_std_1 = self.log_std
        std_1 = log_std_1.exp()
        std_2 = log_std_2.exp()
#        import pdb; pdb.set_trace()
        kl = torch.mean(log_std_2-log_std_1 + (std_1.pow(2)-std_2.pow(2) + (mu_1-mu_2).pow(2))/(2 * std_2.pow(2) + 1e-6))
        return kl  

def train_model(posterior_model, prior_model, train_loader, n_epochs, criterion, optimizer, empirical_density, alpha):
    '''
    This training function is assumed to be used exclusively in an active learning setting.
    It means that we don't need to fix the batch size as we are never going to learn on more than 1000 examples
    '''
    
    # to track the training log likelihood as the model trains
    train_loss = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_loss = [] 
    
    # track the kl loss
    kl_loss = []
    avg_kl_loss = []
    
    # track parameters
    mu_model = []
    sigma_model = []
    
    mu_prior = torch.distributions.Normal(0,10).sample()
    std_prior = torch.distributions.Gamma(0.001, 0.001).sample()
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################        
        for _, (batch_x) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            batch_x = batch_x.to(device)  
            
            mu, sigma = posterior_model()
            kl = posterior_model.KL_weight(mu_prior, std_prior)
            
            if criterion.__name__ == 'empirical_log_loss':
                empirical_loss = criterion(mu, sigma , batch_x)
            elif criterion.__name__ == 'log_loss':
                empirical_loss = criterion(mu, sigma , batch_x)
            elif (criterion.__name__ == 'total_variation_loss') or (criterion.__name__ =='hellinger_loss'):
#                import pdb; pdb.set_trace() 
                empirical_loss = criterion(empirical_density, batch_x, mu, sigma)
            elif criterion.__name__ == 'total_variation_loss':
                empirical_loss = criterion(empirical_density, batch_x, mu, sigma)
            else:
#                import pdb; pdb.set_trace()                 
                empirical_loss = criterion(empirical_density, alpha, mu, sigma, batch_x)
#            import pdb; pdb.set_trace()        
            loss = empirical_loss #+ kl
              
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()                 
            # perform a single optimization step (parameter update)
            optimizer.step() 
            # record training log likelihood, KL and accuracy
            train_loss.append(empirical_loss.item()) 
            kl_loss.append(kl.item())

        # Get descriptive statistics of the training log likelihood, the training accuracy and the KL over MC_sample
        train_batch = np.average(train_loss)   
        kl_batch = np.average(kl_loss)                
        
        # Store the descriptive statistics to display the learning behavior 
        avg_train_loss.append(train_batch)
        avg_kl_loss.append(kl_batch)
        
        mu, sigma = posterior_model()
        mu_model.append(mu.item())
        sigma_model.append(sigma.item())
        # print training/validation statistics 
        pbar.set_postfix(train_loss=train_batch, kl_loss = kl, mu = mu.item(), sigma=sigma.item())
        
        # clear lists to track the monte carlo estimation for the next epoch
        train_loss = []                       
                         
    return  posterior_model, avg_train_loss , avg_kl_loss, mu_model, sigma_model

torch.pi = torch.tensor(np.pi, dtype=torch.float64)
def gauss(mu, sigma, x):
    return 1/(sigma * torch.sqrt(2. * torch.pi)) * torch.exp( - (x - mu)**2. / (2. * sigma**2.)) + 1e-30


def gauss_np(mu, sigma, x):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

## TODO: write higher level loss functions
def log_loss(*args):
    mu, sigma, batch_x = args 
#    import pdb; pdb.set_trace()
    pdf = gauss(mu, sigma, batch_x)
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return -pdf.log().mean()

def empirical_log_loss(*args):
    mu, sigma, batch_x = args 
#    import pdb; pdb.set_trace()
    loss = ((batch_x-mu)**2/(2* sigma**2)) + sigma.log()
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return loss.mean()

def total_variation_loss(*args):
    empirical_density, batch_x, mu, sigma = args 
    loss =  0.5 * torch.abs(1-gauss(mu, sigma, batch_x)/torch.tensor(empirical_density(batch_x)))
    return loss.mean()

def hellinger_loss(*args):
    empirical_density, batch_x, mu, sigma = args 
    loss = 0.5 * (1-torch.sqrt(gauss(mu, sigma, batch_x)/torch.tensor(empirical_density(batch_x)) + 1e-12))**2
#    import pdb; pdb.set_trace()
    return loss.mean()

# I made a mistake here with the denominator. It should be 1/ (alpha*(1-alpha)
def alpha_loss(*args):
    empirical_density, alpha, mu, sigma, batch_x = args 
    loss =  (1/ (alpha*(1-alpha))) * (1-gauss(mu, sigma, batch_x)**(1-alpha) * (torch.tensor(empirical_density(batch_x))**(alpha-1)))
    return loss.mean()

from scipy.stats import gaussian_kde
kernel = gaussian_kde(g)

batch_sample = 1024
train_dataset = MyData(data=g)
trainLoader = DataLoader(train_dataset, batch_size=batch_sample)       
priorLayer = GaussianLayerLRT(shape=(1,1), init="prior")  
for params in priorLayer.parameters():
  params.requires_grad=False 
      
def train_with_loss(crit, n_epochs = 200):
    gaussLayer = GaussianLayerLRT(shape=(1,1), init="standard")        
    optimizer = torch.optim.Adam(gaussLayer.parameters(), lr=0.1)
    _, training_loss, kl_loss, mu_model, sigma_model  = train_model(gaussLayer, priorLayer, trainLoader, n_epochs, crit, optimizer, kernel, alpha=0.75)

    ## Analyze the learned parameters
    mu, std = gaussLayer()
    mu = mu.detach().numpy()[0]
    std = std.detach().numpy()[0]
    return mu, std, training_loss, kl_loss, mu_model, sigma_model


n_epochs = 200
mu_em, std_em, training_loss_em, kl_loss_em, _, _= train_with_loss(empirical_log_loss, n_epochs)
mu_kl, std_kl, training_loss_kl, kl_loss_kl, _, _ = train_with_loss(log_loss, n_epochs)
mu_tv, std_tv, training_loss_tv, kl_loss_tv, _, _ = train_with_loss(total_variation_loss, n_epochs )
mu_hl, std_hl, training_loss_hl, kl_loss_hl, mu_model, sigma_model = train_with_loss(hellinger_loss, n_epochs)
mu_a, std_a, training_loss_a, kl_loss_a, mu_model, sigma_model  = train_with_loss(alpha_loss, n_epochs)

plt.plot(training_loss_em)
plt.plot(kl_loss)

min_x = g.min()
max_x = g.max()
bound = np.max((np.abs(min_x), np.abs(max_x)))

x = np.linspace(min_x, bound, 100)

sns.distplot(g, bins=100)
#plt.plot(x, gauss_np(mu_em, std_em, x), label="analytical kl")
plt.plot(x, gauss_np(mu_kl, std_kl, x), label="kl")
plt.plot(x, gauss_np(mu_tv, std_tv, x), label="total variation")
plt.plot(x, gauss_np(mu_hl, std_hl, x), label="hellinger")
plt.plot(x, gauss_np(mu_a, std_a, x), label="alpha")
plt.legend()
plt.title("Losses derived from robust divergences")
plt.savefig("robust_losses_2", dpi=300)

