#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:03:26 2018

@author: geoffrey
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import neurone as ne
import source
import activation_function



perceptron = ne.neurone(1)

dataset =[[1,2,1],
          [2.5,-3,0],
          [-4,3.5,0],
          [2,4,1],
          [-4,6,1],
          [4,4,1],
          [-1,0.75,0],
          [2.5,0,1],
          [3,0.5,1],
          [-2,-1,0],
          [0.5,-2.5,0],
          [5,0,1],
          [0.5,1.5,1],
          [-0.5,2,1],
          [-4,5.5,1],
          [-2,1.75,0],
          [-3,-1,0],
          [-1,3,1],
          [-1,2.25,1],
          [-0.5,-0.5,0],
          [-1,-1,0],
          [4,4,1],
          [5,-6,0],
          [6,-6.5,0],
          [3,-4,0],
          [1,6,1],
          [1,5,1],
          [-4,-0.5,0],
          [5,-5.25,0],
          [-1,0.5,0],
          [-1,5,1],
          [-2,5.5,1],
          [-3,7,1],
          [-7,-4,0],
          [5,-1,1],
          [2,2,1],
          [-1.5,2.25,0],
          [5,-5.5,0],  
          [-2,-3,0],
          [-3,2.75,0],
          [0.25,1.25,1],
          [3,0,1],
          [4,-1,1],
          [-7,5.56,0],
          [-1.5,1.46,0],
          [0.07,0.94,1],
          [3.32,-3.33,0],
          [-4.78,3.5,0],
          [2.21,-1.16,1],
          [-4.97,6,1]]

data = np.array(dataset)


#source.plot_2d(data)
"""w = [0.1,1,-1]
for row in dataset:
    x = perceptron.prediction(w,row)
    print("Expected=%d, Predicted=%d" % (row[-1], x))"""
    
    
    
w = perceptron.train_weight(data,0.1,activation_function.Heavyside)
print(w)





















