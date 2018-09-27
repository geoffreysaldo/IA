#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:08:17 2018

@author: geoffrey
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import sourcec
import activation_function

class neurone:
    """ class defining a neuron characterised by:
    - Number of input """
    
    def __init__(self,nbr_input):
        # constructor of the class
        self.numberInput = nbr_input

    

    def prediction(self,weight,row,activation):
        output = weight[0]
        for i in range(len(row)-1):
            output += weight[i+1]*row[i]
        return activation(output)
    
    
    def basic_train_weight(self,data_train,learning_rate,activation):
        weights = np.array([0.0 for i in range(len(data_train[0]))])
        for turn in range(15):
            sum_error =0
            for row in data_train:
                error = row[-1] - self.prediction(weights,row,activation)
                sum_error += error**2
                weights[0] = weights[0] + learning_rate * error
                for i in range(len(row)-1):
                    weights[i+1] = weights[i+1] + learning_rate * error * row[i]
            print('>turn=%d, lrate=%.3f, error=%.3f' % (turn, learning_rate, sum_error))
        return np.array(weights)

    
    def basic_train_weight_with_history(self,data_train,learning_rate,activation):
        weights_history =[]
        weights = np.array([0.0 for i in range(len(data_train[0]))])
        for turn in range(15):
            sum_error =0
            for row in data_train:
                error = row[-1] - self.prediction(weights,row,activation)
                sum_error += error**2
                weights[0] = weights[0] + learning_rate * error
                for i in range(len(row)-1):
                    weights[i+1] = weights[i+1] + learning_rate * error * row[i]  
            weights_history.append([weights[0],weights[1],weights[2]])  
            print('>turn=%d, lrate=%.3f, error=%.3f' % (turn, learning_rate, sum_error))
        return np.array(weights_history)






















