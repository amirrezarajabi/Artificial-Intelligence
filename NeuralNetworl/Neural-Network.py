import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image 
from numpy import asarray
def activation_function(x, i):
    if(i == 0):         #sigmoid
        return 1/(1 + np.exp(-x))
    if(i == 1):            #tanh
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    if(i == 2):            #relu
        return max(0, x)
    if(i == 3):       #leakyrelu
        return max(0.01 * x, x)
    
def activation_function_p(x, i):
    if(i == 0):         #sigmoid
        return activation_function(x, 0)*(1-activation_function(x,0))
    if(i == 1):            #tanh
        return 1-activation_function(x,1)*activation_function(x,1)
    if(i == 2):            #relu
        if(x >= 0):
            return 1
        return 0
    if(i == 3):       #leakyrelu
        if(x >= 0):
            return 1
        return 0.01

class NeuralNetwork:
    def __init__(self, X, y, number_of_layers, activation_list, iters, alpha = 0.01):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.num_layers = number_of_layers
        self.w = [0]
        self.b = [0]
        self.A = [0]
        self.Z = [0]
        self.dw = [0]
        self.db = [0]
        self.dz = [0]
        self.iters = iters
        self.m = y.shape[1]
        self.activations = activation_list
        
        self.w.append(np.random.randn(self.num_layers[0], self.X.shape[0]))
        self.b.append(np.zeros((self.num_layers[0], 1)))
        self.dw.append(np.random.randn(self.num_layers[0], self.X.shape[0]))
        self.db.append(np.zeros((self.num_layers[0], 1)))

        for i in range(1, len(self.num_layers)):
            self.w.append(np.random.randn(self.num_layers[i], self.num_layers[i - 1]))
            self.b.append(np.zeros((self.num_layers[i], 1)))
            self.dw.append(np.random.randn(self.num_layers[i], self.num_layers[i - 1]))
            self.db.append(np.zeros((self.num_layers[i], 1)))
        
        for i in range(len(self.num_layers)):
            self.Z.append(np.zeros((self.num_layers[i], self.m)))
            self.A.append(np.zeros((self.num_layers[i], self.m)))
            self.dz.append(np.zeros((self.num_layers[i], self.m)))

    def forward_propagation(self):
        self.Z[1] = np.dot(self.w[1], self.X) + self.b[1]
        self.A[1] = activation_function(self.Z[1], self.activations[0])
        for i in range(2, len(self.A)):
            self.Z[i] = np.dot(self.w[i], self.A[i - 1]) + self.b[i]
            self.A[i] = activation_function(self.Z[i], self.activations[i-1])

    def predict(self, x):
        self.Z[1] = np.dot(self.w[1], x) + self.b[1]
        self.A[1] = activation_function(self.Z[1], self.activations[0])
        for i in range(2, len(self.A)):
            self.Z[i] = np.dot(self.w[i], self.A[i - 1]) + self.b[i]
            self.A[i] = activation_function(self.Z[i], self.activations[i-1])

    def backward_propagation(self):
        k = len(self.dz) - 1
        self.dz[k] = self.A[k] - self.y
        self.dw[k] = 1/self.m * np.dot(self.dz[k], self.A[k - 1].T)
        self.db[k] = 1/self.m * np.sum(self.dz[k], axis=1, keepdims=True)
        k = k - 1
        while(k > 1):
            self.dz[k] = np.dot(self.w[k + 1].T, self.dz[k + 1]) * activation_function_p(self.Z[k], self.activations[k])
            self.dw[k] = 1/self.m * np.dot(self.dz[k], self.A[k - 1].T)
            self.db[k] = 1/self.m * np.sum(self.dz[k], axis=1, keepdims=True)
            k = k - 1
        self.dz[k] = np.dot(self.w[k + 1].T, self.dz[k + 1]) * activation_function_p(self.Z[k], self.activations[k])
        self.dw[k] = 1/self.m * np.dot(self.dz[k], self.X.T)
        self.db[k] = 1/self.m * np.sum(self.dz[k], axis=1, keepdims=True)
        for i in range(1, len(self.w)):
            self.w[i] = self.w[i] - self.alpha * self.dw[i]
            self.b[i] = self.b[i] - self.alpha * self.db[i]
        

    def gd(self):
        for i in range(self.iters):
            self.forward_propagation()
            self.backward_propagation()

numbers = [3, 2] #neural network architecture:input shape + numbers + output shape
activ = [0, 0] 
activ.append(0)
numbers.append(y.shape[0])
nn = NeuralNetwork(X, y, numbers, activ, 100000)

### /\ /V\ [] [Z [Z (≡ ¯/_ /\ [Z /\ _] /\ !3 [] ###
