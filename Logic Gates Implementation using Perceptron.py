#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd


# In[2]:


class Perceptron:
    def __init__(self, eta, epochs, activationFunction):
        self.weights = np.random.randn(3) * 1e-4
        print(f"self.weights: {self.weights}")
        self.eta = eta
        self.epochs = epochs
        self.activationFunction = activationFunction

    def fit(self, X, y):
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # concactination
        print(f"X_with_bias: \n{X_with_bias}")

        for epoch in range(self.epochs):
            print(f"for epoch: {epoch}")
            y_hat = self.activationFunction(X_with_bias, self.weights)
            print(f"predicted value: \n{y_hat}")
            error = self.y - y_hat
            print(f"error: \n{error}")
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)
            print(f"updated weights: \n{self.weights}")
            print("#############\n")

    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones((len(self.X), 1))]
        return self.activationFunction(X_with_bias, self.weights)


# In[3]:


activationFunction = lambda inputs, weights: np.where(np.dot(inputs, weights) > 0 , 1, 0)


# # AND

# In[4]:


data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,0,0,1]}

AND = pd.DataFrame(data)
AND


# In[5]:


X = AND.drop("y", axis=1) # axis = 1 >>> dropping accross column
X


# In[6]:


y = AND['y']
y.to_frame()


# In[7]:


model = Perceptron(eta = 0.5, epochs=10, activationFunction=activationFunction)


# In[8]:


model.fit(X,y)


# In[9]:


model.predict(X)


# # OR

# In[10]:


data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,1,1,1]}

OR = pd.DataFrame(data)
OR


# In[11]:


X = OR.drop("y", axis=1) # axis = 1 >>> dropping accross column
X


# In[12]:


y = OR['y']
y.to_frame()


# In[13]:


model = Perceptron(eta = 0.5, epochs=10, activationFunction=activationFunction)


# In[14]:


model.fit(X,y)


# # XOR

# In[15]:


data = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,1,1,0]}

XOR = pd.DataFrame(data)
XOR


# In[16]:


X = XOR.drop("y", axis=1) # axis = 1 >>> dropping accross column
X


# In[17]:


y = XOR['y']
y.to_frame()


# In[18]:


model = Perceptron(eta = 0.5, epochs=50, activationFunction=activationFunction)


# In[19]:


model.fit(X,y)


# In[20]:


model.predict(X) # Doesn't predict since its a non-linear problem


# In[21]:


XOR.plot(kind="scatter", x="x1", y="x2", c="y", s=500, cmap="winter")


# In[22]:


# Cannot segregate since non-linear data
# Hence MLPs are used

