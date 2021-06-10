#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:40:26 2021

@author: root
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n_sample = 1000
mixture = np.random.binomial(n_sample, 0.99)
g1 = np.random.normal(0,1, mixture)
g2 = np.random.normal(5,5, n_sample-mixture)

np.random.normal()
g = np.concatenate([g1,g2])

ax = sns.distplot(g,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})

from scipy.stats import gaussian_kde
kernel = gaussian_kde(g)
min_x = g.min()
max_x = g.max()

grid = np.linspace(min_x, max_x, 100)
est = kernel(grid)

sns.distplot(g, bins=100)
plt.plot(grid, est)
