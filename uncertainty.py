#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:23:12 2021

@author: root
"""

from utils_two_moons import predict_model, MyData
import numpy as np
from torch.utils.data import DataLoader


def compute_entropy(p):
    '''
    Computes the entropy of a categorical probability distribution
    It handle the case where one of the class has probability 1. and all the other have 0.
    It sets the convention: 0 * log(0) = 0.
    The function can handle 
    Args:
      p: Float array of any dimension (:, ... ,:, d). The last dimension in each case must be a probability distribution
         i.e it must sums to 1.
    Return
      entropy: Float. The entropy of the probability distribution p.
    '''    
    zero_probs = p == 0
    log_p = np.log(p)
    log_p[zero_probs] = 0    
    entropy = -(p * log_p).sum(-1)  
    return entropy

def compute_predictive_uncertainty(model, loader, MC_sample, no_classes):  
  """
  Computes the predictive entropy H(y|x,D)
  """
  probs = predict_model(model, loader, MC_sample, no_classes)
  batch_size, _ = probs.shape
  predictive_uncertainty = compute_entropy(probs)  
  assert predictive_uncertainty.shape == (batch_size,)            
  return predictive_uncertainty


def retain_data(data, retain, mode):
    """    
    Sort the data, and return a percentage in ascending or descending order.
    This function is intended to be used with entropies or probabilities. In which
    case we would retain data points with lowest entropies, or highest probabilities.
    Args:
        data: np array of unordered value
        retain: Float number between 0 and 1, refer to the percentage
        of certain data to keep.
    Return:
        cerain_data: most certain retained data
    """
    if not isinstance(data, (np.ndarray,)):
        data = np.array(data) 
    # Sort data
    sorted_id = np.argsort(data)
    batch_size = data.shape[0]
    truncate = int(batch_size * retain)
    if mode=="ascending":
        certain_id = sorted_id[:truncate]
    elif mode=="descending":
        certain_id = sorted_id[::-1][:truncate]
    else:
        raise Exception("mode should be ascending or descending")
    return certain_id

def sample_lowest_entropy(retain, model, data, labels, MC_sample, no_classes):
    '''
    Sample a percentage of data with the highest certainty 
    Args:
      retain: Float number between 0 and 1, refer to the percentage
      model: predictive function       
      data: Pool of data.
      labels: Pool of labels
    Return
        low_entropy_data: the retained data with the lowest entropy
        low_entropy_labels: labels associated with retained data with the lowest entropy
    '''    
    # We need the loader only to use predictions
    loader = DataLoader(MyData(data, labels), batch_size=512, shuffle=False)
    predictions = predict_model(model, loader, MC_sample, no_classes)            
    # import pdb;pdb.set_trace()
    entropies = compute_entropy(predictions).squeeze()
    low_entropies_id = retain_data(entropies, retain, "ascending")
#    import pdb; pdb.set_trace()
    low_entropy_data = data[low_entropies_id]
    low_entropy_labels = labels[low_entropies_id]
    new_loader = DataLoader(MyData(low_entropy_data, low_entropy_labels), batch_size=512, shuffle=False)
    return new_loader

def sample_highest_density(retain, kde, pca, data, labels):
    """
    Sample a percentage of data with the highest probability score 
    Args:
      retain: Float number between 0 and 1, refer to the percentage
      kde, pca: density functions
      data: Pool of data.
      labels: Pool of labels
    Return
        low_entropy_data: the retained data with the lowest entropy
        low_entropy_labels: labels associated with retained data with the lowest entropy
    """  
    log_probabilities = kde.score_samples(pca.transform(data))
    probabilities = np.power(np.exp(1), log_probabilities)
    high_probabilities_id = retain_data(probabilities, retain, "descending")
    high_probabilities_data = data[high_probabilities_id]
    high_probabilities_labels = labels[high_probabilities_id]
    new_loader = DataLoader(MyData(high_probabilities_data, high_probabilities_labels), batch_size=512, shuffle=False)
    return new_loader


def sample_lowest_entropy_highest_density(retain, model, kde, pca, data, labels, MC_sample, no_classes):
    '''
    Sample a percentage of data with the highest certainty and the lowest entropy.
    Since we want to sample in different mode ascending for entropy and 
    descending for density, we use -H(y|x)p(x) as our metric so that they
    we can just take the highest values
    Args:
      retain: Float number between 0 and 1, refer to the percentage
      model: predictive function       
      data: Pool of data.
      labels: Pool of labels
    Return
        low_entropy_high_density_data: the retained data with the lowest entropy
        low_entropy_high_density_labels: labels associated with retained data with the lowest entropy
    '''    
    # We need the loader only to use predictions
    loader = DataLoader(MyData(data, labels), batch_size=512, shuffle=False)
    predictions = predict_model(model, loader, MC_sample, no_classes)            
    # import pdb;pdb.set_trace()
    entropies = compute_entropy(predictions).squeeze()
    log_probabilities = kde.score_samples(pca.transform(data))
    probabilities = np.power(np.exp(1), log_probabilities)
    entropy_density = (1/entropies+ 1e-6) *  probabilities
    low_entropies_high_density_id = retain_data(entropy_density, retain, "descending")
#    import pdb; pdb.set_trace()
    low_entropy_high_density_data = data[low_entropies_high_density_id]
    low_entropy_high_density_labels = labels[low_entropies_high_density_id]
    new_loader = DataLoader(MyData(low_entropy_high_density_data, low_entropy_high_density_labels), batch_size=512, shuffle=False)
    return new_loader
