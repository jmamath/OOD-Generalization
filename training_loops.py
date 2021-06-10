#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 07:01:52 2021

@author: root
"""
from tqdm import trange
import torch 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

def train_model_robust_losses(model, base_model, train_loader, n_epochs, criterion, optimizer):
    '''
    This training function is assumed to be used exclusively in an active learning setting.
    It means that we don't need to fix the batch size as we are never going to learn on more than 1000 examples
    FGSM: https://github.com/locuslab/fast_adversarial/blob/master/MNIST/train_mnist.py
    Mixup: https://github.com/leehomyc/mixup_pytorch/blob/master/main_cifar10.py
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
            if (criterion.__name__ == "log_loss") or (criterion.__name__ == "empirical_log_loss"):
                loss = criterion(prediction, batch_y)  
            else:
                cond_density = torch.sigmoid(base_model(batch_x))
                loss = criterion(prediction, batch_y, cond_density)
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

def compute_svi_pred(posterior_model ,batch_x, MC_sample, no_classes): 
  # compute MC_sample Monte Carlo predictions
  predictions = torch.zeros((MC_sample,len(batch_x), no_classes), dtype=torch.float32).to(device)
  for j in range(MC_sample):
    # forward pass: compute predicted outputs by passing inputs to the model
    predictions[j] = posterior_model(batch_x)      
  pred = predictions.mean(0)   
  return pred

def train_model_dropout(posterior_model, prior_model, MC_sample, train_loader, n_epochs, criterion, optimizer, no_classes, mixup=False, fgsm=False, kde=None, pca=None):
    '''
    This training function is assumed to be used exclusively in an active learning setting.
    It means that we don't need to fix the batch size as we are never going to learn on more than 1000 examples
    '''
    
    # to track the training log likelihood as the model trains
    train_log_likelihood = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_log_likelihood = [] 
    
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################        
        for _, (batch_x, batch_y) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if (mixup == True):
                batch_x, batch_y = shuffle_minibatch(batch_x, batch_y, no_classes, True)
            if fgsm == True:
                epsilon=0.3
                alpha = 0.375
                delta = torch.zeros_like(batch_x).uniform_(-epsilon, epsilon).to(device)
                delta.requires_grad = True
                output = posterior_model(batch_x + delta)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = torch.max(torch.min(1-batch_x, delta.data), 0-batch_x)
                delta = delta.detach()
            batch_x = batch_x.to(device)           
            batch_y = batch_y.to(device)
#            import pdb; pdb.set_trace()
            # compute MC_sample Monte Carlo predictions       
            ## We perform many sample
            if kde is not None:
                p_x = kde.score_samples(pca.transform(batch_x))
                p_x = torch.tensor(np.power(np.exp(1), p_x)).unsqueeze(1)
                prediction = compute_svi_pred(posterior_model, batch_x, MC_sample, no_classes)                 
#                import pdb; pdb.set_trace()
                log_likelihood = criterion(p_x*prediction, batch_y)
            
            prediction = compute_svi_pred(posterior_model, batch_x, MC_sample, no_classes)                 
            log_likelihood = criterion(prediction, batch_y)
            # import pdb; pdb.set_trace()
            loss = log_likelihood
           # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()                 
            # perform a single optimization step (parameter update)
            optimizer.step() 
            # record training log likelihood, KL and accuracy
            train_log_likelihood.append(log_likelihood.item())          

        # Get descriptive statistics of the training log likelihood, the training accuracy and the KL over MC_sample
        train_log_lik = np.average(train_log_likelihood)                   
        
        # Store the descriptive statistics to display the learning behavior 
        avg_train_log_likelihood.append(train_log_lik)
        
        
        # print training/validation statistics 
        pbar.set_postfix(train_log_likelihood=train_log_lik)
        
        # clear lists to track the monte carlo estimation for the next epoch
        train_log_likelihood = []                       
        # if epoch % 20 == 0:
        #   print("Saving model at epoch ", epoch)
        #   torch.save(posterior_model.state_dict(), './'+'{}_state_{}.pt'.format(model.name, epoch))              
                  
    return  posterior_model, avg_train_log_likelihood

def shuffle_minibatch(inputs, targets, no_classes, mixup=True):
    """Shuffle a minibatch and do linear interpolation between couple of inputs
    and targets.
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]
    
    # We start by generating a batch of couple of inputs and targets.
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, no_classes)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, no_classes)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)
    
    # For each couple of data, we have an associated alpha for mixing up.
    if mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

#    b = np.tile(a[..., None, None], [1, 3, 32, 32]) for images of shape [3,32,32]
    b = np.tile(a[...], [2])    # We adapt to the toy dataset of shape [2]

    # We compute the linear interpolation between inputs x hat
    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()
    inputs_shuffle = inputs1 + inputs2
    
    # We prepare the interpolation for each class so one dimension per class
    c = np.tile(a, [1, no_classes])

    # We compute the linear interpolation for each dimension of the labels
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()


    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle

def train_model_env(posterior_model, prior_model, MC_sample, env_loader, n_epochs, criterion, optimizer, algo=None):
    '''
    Here, we make the assumption that each batch is one environment, and we suppose that data in this environment
    fits into memory . But we could filter the batch size in MyData and assuming that 
    dataloader will always output one env at a time.
    '''
    
    # to track the training loss and penalty
    train_loss = []
    train_penalty = []    
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################          
        for i, (env_e_x, env_e_y) in enumerate(env_loader, 1):
            # Here we use a torch tensor to store the evolution of the loss and the penalty
            # for IRM and VREX, we have chosen this way to harmonize how the final OOD risk is computed
            nb_env = env_loader.dataset.data.shape[0]
            loss = torch.zeros(nb_env, dtype=torch.float32).to(device)
            penalty = torch.zeros(nb_env, dtype=torch.float32).to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()            
            env_e_x = env_e_x[0].to(device)           
            env_e_y = env_e_y[0].to(device)

            prediction = posterior_model(env_e_x)
            if algo=="IRM":
                env_risk = criterion(dummy_w*prediction, env_e_y)
                grad_env_error = grad(env_risk, dummy_w, create_graph=True)[0]
    #            import pdb; pdb.set_trace()            
                loss[i-1] = env_risk
                penalty[i-1] = grad_env_error**2
            elif algo=="VREX":
                env_risk = criterion(prediction, env_e_y)
                loss[i-1] = env_risk
                penalty[i-1] = (env_risk)   
            else:
                raise Exception("Only IRM and VREX are admitted")         
        # backward pass: compute gradient of the loss with respect to model parameters
        if algo == "IRM":
            loss = loss.sum(0)
            penalty = penalty.sum(0)
            (loss.sum(0) + penalty.sum(0)).backward()                 
        elif algo == "VREX":
            loss = loss.sum(0)
            penalty = penalty.var()
            (loss + penalty).backward()                 
        # perform a single optimization step (parameter update)
        optimizer.step()                          
        # Store the descriptive statistics to display the learning behavior 
        train_loss.append(loss.item())
        train_penalty.append(penalty.item())                
        # print training/validation statistics 
        pbar.set_postfix(loss=loss.item(), penalty=penalty.item())
        
        # clear lists to track the monte carlo estimation for the next epoch                      
        # if epoch % 20 == 0:
        #   print("Saving model at epoch ", epoch)
        #   torch.save(posterior_model.state_dict(), './'+'{}_state_{}.pt'.format(model.name, epoch))              
                  
    return  posterior_model, train_loss, train_penalty



