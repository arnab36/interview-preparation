# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 07:36:39 2022

@author: 01927Z744


"""

import numpy as np
import matplotlib.pyplot as plt



pos = 0
scale = 5
size=  20000

# pos=center of distribution, scale=Standard deviation (spread or “width”) of the distribution,
# size= number of samples
values = np.random.normal(pos, scale,size)

# hist(values,bins). bins are like buckets i.e the number of lines you want
plt.hist(values,1000)

plt.show()


