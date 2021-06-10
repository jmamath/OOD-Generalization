#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:18:41 2021

@author: root
"""

import torch
from torch.autograd import grad

def compute_penalty(losses, dummy_w):
    # take the non odd indices
    import pdb; pdb.set_trace()
    g1 = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0] 
    # take the odd indices
    g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0] 
    return (g1 * g2).sum()

#def penalty(logits, y):
#    if use_cuda:
#      scale = torch.tensor(1.).cuda().requires_grad_()
#    else:
#      scale = torch.tensor(1.).requires_grad_()
#    loss = mean_nll(logits * scale, y)
#    grad = autograd.grad(loss, [scale], create_graph=True)[0]
#    return torch.sum(grad**2)

def example_1(n=10000, d=2, env=1):
    x = torch.randn(n, d) * env
    y = x + torch.randn(n, d) * env
    z = y + torch.randn(n, d)
    return torch.cat((x, z), 1), y.sum(1, keepdim=True)

phi = torch.nn.Parameter(torch.ones(4, 1)) 
dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))
opt = torch.optim.SGD([phi], lr=1e-3) 
mse = torch.nn.MSELoss(reduction="none")

environments = [example_1(env=0.1), example_1(env=1.0)]

for iteration in range(50000): 
    error = 0
    penalty = 0
    for x_e, y_e in environments:
        p = torch.randperm(len(x_e))
        error_e = mse(x_e[p] @ phi * dummy_w, y_e[p]) 
        penalty += compute_penalty(error_e, dummy_w) 
#        import pdb; pdb.set_trace()
        # The total error is the sum of error made on each environment
        error += error_e.mean()
    opt.zero_grad()
    (1e-5 * error + penalty).backward() 
    opt.step()
    if iteration % 1000 == 0: print(phi)