#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:16:55 2021

@author: root
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import trange

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

################# MODEL AND DATA #################
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
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1,
                 name="Standard"):
        """
        You should definitely mind the features transformation depending on the dataset,
        it might be different between MNIST and CIFAR-10 for instance,
        bayesLinear1 will have respectively 16*4*4 and 16*5*5 in_features.
        """
        super(NeuralNet, self).__init__()        
        self.Linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Linear2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):              
        x = F.relu(self.Linear1(x))           
        x = self.Linear2(x)
        return x  
    
class EnsembleNeuralNet(nn.Module):
    def __init__(self, base_neural_nets,
                 name="Standard_Item"):
        """
        You should definitely mind the features transformation depending on the dataset,
        it might be different between MNIST and CIFAR-10 for instance,
        bayesLinear1 will have respectively 16*4*4 and 16*5*5 in_features.
        """
        super(EnsembleNeuralNet, self).__init__() 
        self.ensemble = nn.ModuleList(base_neural_nets)
        
    def forward(self, x):              
        for i, l in enumerate(self.ensemble):
            pred = self.ensemble[i](x)
        return pred/len(self.ensemble)
    
class MCDropout(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1,
                 name="Dropout"):
        """
        You should definitely mind the features transformation depending on the dataset,
        it might be different between MNIST and CIFAR-10 for instance,
        bayesLinear1 will have respectively 16*4*4 and 16*5*5 in_features.
        """
        super(MCDropout, self).__init__()        
        self.Linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.Linear2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):              
        x = F.relu(self.Linear1(x))   
        x = self.dropout(x)                
        x = self.Linear2(x)                   
        return x      
    
def mixup_log_loss(prediction, label):
#    import pdb; pdb.set_trace()
    loss = nn.LogSoftmax()
#    import pdb; pdb.set_trace()
    log_loss = loss(prediction) * label
#    zero_probs = pdf == 0
#    pdf[zero_probs] = 1e-6
    return -log_loss.mean()    
    
def compute_accuracy(pred, y):
  _, predicted = torch.max(F.softmax(pred), 1)
  total = len(pred)
  correct = (predicted == y).sum()
  accuracy = 100 * correct.cpu().numpy() / total 
  return accuracy  

def evaluate_model(model, loader, MC_sample, no_classes):
  with torch.no_grad():
      acc_final = []
      for x, y in loader: # batch_level   
        x = x.to(device)
        y = y.to(device)   
        predictions = torch.zeros(MC_sample,len(x),no_classes).to(device)
        for j in range(MC_sample):
          predictions[j] = model(x)    
        pred = predictions.mean(0) 
        accuracy = compute_accuracy(pred, y)
        acc_final.append(accuracy)
  return np.array(acc_final).mean()  

def predict_model(model, loader, MC_sample, no_classes):
  with torch.no_grad():
    predictions = []
    for x, y in loader: # batch_level      
      x = x.to(device)  
      pred = torch.zeros(MC_sample,len(x), no_classes).to(device)    
      for j in range(MC_sample):      
        pred[j] = F.softmax(model(x))    
      pred_batch = pred.mean(0)  
      predictions.append(pred_batch.detach().cpu().numpy())
  return np.concatenate(predictions)


################# CALIBRATION #################
def to_onehot(y, out_dim):
  """
  Transforms an array of Long to its one hot representation
  out_dim is the number of class represented in the array.
  """
  y_onehot = torch.FloatTensor(len(y), out_dim).to(device)
  y_onehot.zero_()
  # import pdb; pdb.set_trace()
  y_onehot.scatter_(1, y.unsqueeze(1), 1)
  return y_onehot

def brier_score(model, loader, MC_sample, no_classes):
  """
  Computes the brier score, for that we need to transform
  the true label in one hot representation, and also transform the
  data to cuda in order to perform computation with cuda BNN.
  """
  brier_score = []
  for x, y in loader:    
    x = x.to(device)
    y = y.to(device)
    y_onehot = to_onehot(y, no_classes)      
    pred = torch.zeros(MC_sample, len(x),no_classes).to(device)
    for j in range(MC_sample):      
      pred[j] = model(x)    
    pred_batch = pred.mean(0)    
    probs = F.softmax(pred_batch)
    # import pdb; pdb.set_trace()
    brier = (probs-y_onehot).pow(2).sum(1).mean(0)
    brier_score.append(brier.detach().cpu().numpy())
  return np.array(brier_score).mean()

#@title Expected Calibration Error
def expectation_calibration_error(model, loader, MC_sample, num_buckets, no_classes):
  """
  The formula can be found in arxiv:1706.04599
  """
  # (1) We need to get the prediction
  ece = []
  total_sizes = []
  for x, y in loader:
    # These are     
    ece_batch = []    
    x = x.to(device)
    y = y.to(device)
    pred = torch.zeros(MC_sample, len(x),no_classes).to(device)
    for j in range(MC_sample):      
      pred[j] = model(x)    
    pred_batch = pred.mean(0)    
    probs, pred = torch.max(F.softmax(pred_batch), 1)
    # (2) Partition the probs to fit in each bin of the segment [1/no_class,1]
    # and get mean accuracy and confidence per bin. We choose 1/no_class to start because no prediction
    # could be lower than that, it is the uniform distribution i.e no prediction can have confidence lower than
    # 0.1 in 10-class classification.
    interval = np.linspace(1/no_classes,1, num_buckets)
    # container to store the accuracy and confidence within each bin of the interval
    acc = []
    conf = []
    bin_size = []
    for i in range(num_buckets-1):
      lower_bin, upper_bin = interval[i], interval[i+1]
      probs_mask = (probs > lower_bin) & (probs <= upper_bin)
      if probs_mask.sum() > 0:
        # we look into the bin only if it is not empty
        probs_bin = probs[probs_mask]
        pred_bin = pred[probs_mask]    
        y_bin = y[probs_mask]
        len_bin = len(probs_bin)
        bin_size.append(len_bin)        
        acc_bin = ((y_bin == pred_bin).sum())/ float(len_bin)  
        acc.append(acc_bin.detach().cpu().numpy())
        conf_bin = probs_bin.sum()/len_bin
        conf.append(conf_bin.detach().cpu().numpy())              
    # This ece_batch is not complete, because we should divide it by the total number of data
    # however we don't have it at the batch level. So we should wait until we 
    # get out of the loop to divide by the total number of sample            
    ece_batch = np.sum(np.array(bin_size) * np.abs(np.array(acc)-np.array(conf)))
    ece.append(ece_batch)    
    total_sizes.append(np.sum(np.array(bin_size)))  
  # import pdb; pdb.set_trace() 
  ece = np.array(ece)
  total_sizes = np.array(total_sizes)
  ece = np.sum(ece) / np.sum(total_sizes)
  return ece

