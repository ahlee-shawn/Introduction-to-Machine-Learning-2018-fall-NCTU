import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    #Write codes here
    activate = lambda i : sigmoid(i)
    vectorized_activate = np.vectorize(activate)
    #bias = np.array(np.ones(m), ndmin=2).transpose()
    bias = np.ones((5000, 1))
    a1 = np.concatenate((bias, X), axis=1)
    z2 = np.dot(a1, theta1.transpose())
    a2 = np.concatenate((bias, vectorized_activate(z2)), axis=1)
    z3 = np.dot(a2, theta2.transpose())
    h = vectorized_activate(z3)
    
    return a1, z2, a2, z3, h

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))
    
    return J
    
# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
   
    #Write codes here
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    Delta1 = np.zeros(theta1.shape)
    Delta2 = np.zeros(theta2.shape) 

    theta1_without_bias = np.delete(theta1, 0, axis = 1)
    theta2_without_bias = np.delete(theta2, 0, axis = 1)

    activate = lambda i : sigmoid(i)
    vectorized_activate = np.vectorize(activate)
    activate_gradient = lambda i : sigmoid_gradient(i)
    vectorized_activate_gradient = np.vectorize(activate_gradient)
    learning_rate_divide_m = lambda i : (i * learning_rate) / m
    vectorized_learning_rate_divide_m = np.vectorize(learning_rate_divide_m)

    for i in range(0, m):
        bias = np.ones((1,1)) #
        a1 = np.concatenate((bias, X[i].transpose()), axis = 0)
        z2 = np.dot(theta1, a1)
        a2 = np.concatenate((bias, vectorized_activate(z2)), axis=0)
        z3 = np.dot(theta2, a2)
        h = vectorized_activate(z3)
        delta3 = np.subtract(h, y[i].reshape(10, 1))
        delta2 = np.multiply(np.dot(theta2_without_bias.transpose(), delta3), vectorized_activate_gradient(z2))
        Delta2 += np.dot(delta3, a2.transpose())
        Delta1 += np.dot(delta2, a1.transpose())

    theta1_gradient = vectorized_learning_rate_divide_m(Delta1)
    theta2_gradient = vectorized_learning_rate_divide_m(Delta2)

    grad = np.concatenate((theta1_gradient.flatten(), theta2_gradient.flatten()), axis = 0)

    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)

    return J, grad
    
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250, 'disp': True})
print(fmin)
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
