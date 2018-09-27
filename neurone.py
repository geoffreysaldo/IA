#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:08:17 2018

@author: geoffrey
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import perceptron_source
import activation_function

class neurone:
    """ class defining a neuron characterised by:
    - Treshold of the transfert function
    - Number of input """
    
    def __init__(self,nbr_input):
        # constructor of the class
        self.numberInput = nbr_input

    

    def prediction(self,weight,row,activation):
        output = weight[0]
        for i in range(len(row)-1):
            output += weight[i+1]*row[i]
        return activation(output)
        """if output > self.treshold :
            return 1
        else :
            return 0"""
    
    
    def train_weight(self,data_train,learning_rate,activation):
        weights = [0.0 for i in range(len(data_train[0]))]
        for turn in range(200):
            sum_error =0
            for row in data_train:
                error = row[-1] - self.prediction(weights,row,activation)
                sum_error += error**2
                weights[0] = weights[0] + learning_rate * error
                for i in range(len(row)-1):
                    weights[i+1] = weights[i+1] + learning_rate * error * row[i]
            print('>turn=%d, lrate=%.3f, error=%.3f' % (turn, learning_rate, sum_error))
        return weights



perceptron = neurone(1)

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



"""w = [0.1,1,-1]
for row in dataset:
    x = perceptron.prediction(w,row)
    print("Expected=%d, Predicted=%d" % (row[-1], x))"""
    
w = perceptron.train_weight(dataset,0.1,activation_function.Heavyside)
print(w)




















