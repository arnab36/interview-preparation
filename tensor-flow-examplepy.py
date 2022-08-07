# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 07:29:05 2022

@author: 01927Z744
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)