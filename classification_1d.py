#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 08:15:36 2021

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import trange

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

class MyData(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        """"Initialization, here display serves to show an example
        if false, it means that we intend to feed the data to a model"""
        self.labels = labels
        self.data = data           

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]            
        # Load data and get label
        y = self.labels[index]                
        return X,y   

class NeuralNet(nn.Module):
    def __init__(self, input_dim=1,  output_dim=1,
                 name="Base"):
        """
        You should definitely mind the features transformation depending on the dataset,
        it might be different between MNIST and CIFAR-10 for instance,
        bayesLinear1 will have respectively 16*4*4 and 16*5*5 in_features.
        """
        super(NeuralNet, self).__init__()        
        self.Linear1 = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):              
        x = F.relu(self.Linear1(x))                         
        return x       
    
def compute_accuracy(pred, y):
  _, predicted = torch.max(F.softmax(pred), 1)
  total = len(pred)
  correct = (predicted == y).sum()
  accuracy = 100 * correct.cpu().numpy() / total 
  return accuracy  

def evaluate_model(model, loader):
  with torch.no_grad():
      acc_final = []
      for x, y in loader: # batch_level   
        x = x.to(device)
        y = y.to(device)   
        pred = model(x)    
        accuracy = compute_accuracy(pred, y)
        acc_final.append(accuracy)
  return np.array(acc_final)   

def train_model(model, train_loader, n_epochs, criterion, optimizer):
    '''
    This training function is assumed to be used exclusively in an active learning setting.
    It means that we don't need to fix the batch size as we are never going to learn on more than 1000 examples
    '''
    
    # to track the training log likelihood as the model trains
    training_loss = []
    # to track the average training log likelihood per epoch as the model trains
    avg_training_loss = [] 
    
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################        
        for _, (batch_x, batch_y) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            batch_x = batch_x.to(device)           
            batch_y = batch_y.to(device)
            # compute MC_sample Monte Carlo predictions
            prediction = model(batch_x)      
            loss = criterion(prediction, batch_y)             
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()                 
            # perform a single optimization step (parameter update)
            optimizer.step() 
            # record training log likelihood, KL and accuracy
            training_loss.append(loss.item())          

        # Get descriptive statistics for the batch
        batch_loss = np.average(training_loss)                   
        
        # Store the descriptive statistics to display the learning behavior 
        avg_training_loss.append(batch_loss)
        
        # print training/validation statistics 
        pbar.set_postfix(training_loss=batch_loss)
        
        # clear lists to track the monte carlo estimation for the next epoch
        training_loss = []                       
        # if epoch % 20 == 0:
        #   print("Saving model at epoch ", epoch)
        #   torch.save(posterior_model.state_dict(), './'+'{}_state_{}.pt'.format(model.name, epoch))              
                  
    return  model, avg_training_loss    
    

### Create the dataset
d1 = np.random.normal(loc=-2, scale=.5 ,size=250)

    
n_sample = 250
mixture = np.random.binomial(n_sample, 0.95)
g1 = np.random.normal(2,.5, mixture)
g2 = np.random.normal(-3,.5, n_sample-mixture)

# One very corrupted
g = np.concatenate([g1,g2])

ax = sns.distplot(g,
                  bins=100,
                  kde=True,
                  hist_kws={"linewidth": 15,'alpha':1})


# One not corrupted
#d2 = np.random.normal(loc=2, scale=.5 ,size=500)

#data_tr = np.concatenate([d1,d2])
data_cr = np.concatenate([d1,g])


ax = sns.distplot(data_cr,
                  bins=100,
                  kde=True,
                  hist_kws={"linewidth": 15,'alpha':1})

#d1 = np.random.normal(loc=-1, scale=.5 ,size=500)
#d2 = np.random.normal(loc=1, scale=.5 ,size=500)
#
#data_ts = np.concatenate([d1,d2])

ax = sns.distplot(data_tr,
                  bins=100,
                  kde=True,
                  hist_kws={"linewidth": 15,'alpha':1})

label_1 = np.ones(250)
label_0 = np.ones(250)
label = np.concatenate([label_0,label_1]).astype(np.int64)


#
#y_tr = (data_tr > 0)*1
#y_ts = (data_ts > 0)*1

data_cr = np.expand_dims(data_cr, 1)
#data_tr = np.expand_dims(data_tr, 1)
#data_ts = np.expand_dims(data_ts, 1)


train_dataset = MyData(data_cr, label)
#test_dataset = MyData(data_ts, y_ts)

batch_sample=1024
trainLoader = DataLoader(train_dataset, batch_size=batch_sample)    
#testLoader = DataLoader(test_dataset, batch_size=batch_sample)  

## 3. Learn a base conditional distribution ##

base_nn = NeuralNet(input_dim=1, output_dim=2).double()
optimizer = torch.optim.Adam(base_nn.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()
MC_sample=1
n_epochs = 150
_, training_loss = train_model(base_nn, trainLoader, n_epochs, crit, optimizer)

plt.plot(training_loss)

evaluate_model(base_nn, trainLoader)
#evaluate_model(base_nn, testLoader)

X = np.linspace(-4,4,500)
Y_pred = torch.sigmoid(base_nn(torch.tensor(X).unsqueeze(1)))

plt.plot(X, Y_pred.detach().numpy())




