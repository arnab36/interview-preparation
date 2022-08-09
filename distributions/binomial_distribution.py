# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 08:52:10 2022

@author: 01927Z744

Binomial distribution is a probability distribution that summarises the likelihood that a 
variable will take one of two independent values under a given set of parameters. 
The distribution is obtained by performing a number of Bernoulli trials.

A Bernoulli trial is assumed to meet each of these criteria :

    - There must be only 2 possible outcomes.

    - Each outcome has a fixed probability of occurring. A success has the probability of p,
        and a failure has the probability of 1 – p.

    - Each trial is completely independent of all others.

    
"""

from scipy.stats import binom
import matplotlib.pyplot as plt

# Number of experiments
n = 10

# Probability of success
p = 0.167

# different values of random variable
r_values = list(range(6))

# mean = n*p,  var = n*p*(1-p)
mean, var = binom.stats(n, p)

# Run this
dist = [binom.pmf(r, n, p) for r in r_values ]

# or this
dist = []
for r in r_values:
    dist.append(binom.pmf(r, n, p))

plt.bar(r_values, dist)
plt.show()









