# This script demonstrates training of a neural network model
# on the NASA airfoil noise dataset using the conjugate gradient
# optimization algorithm implemented in minimize.py
#
# The neural network parameters are initialized randomly so each
# run will produce a different model
#
import numpy as np
import matplotlib.pyplot as plt
from minimize import minimize
from loss import loss
from model_predict import model_predict

# Load training and testing datasets
print('Loading airfoil noise training and test dataset...')
training_data = np.loadtxt("airfoil_training_data.dat")  # 1000 points
testing_data = np.loadtxt("airfoil_testing_data.dat")    # 503 points

# In this dataset, we have 5 inputs and one output. Let's split for clarity
X_train = training_data[:, :5]
y_train = training_data[:, 5]
X_test = testing_data[:, :5]
y_test = testing_data[:, 5]

print(f'Data split: {len(y_train)} training, {len(y_test)} test samples')

# Normalize the inputs in X_train and X_test using mean and std of training data 
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

## Initialize model parameters
print('Initializing neural network parameters...')

# We will approximate the airfoil noise as a function of the
# 5 inputs using a simple feedforward neural network model
# of the form
# y(x) = W2*tanh(W1*x + b1) + b2
# where x is a vector of length d, W1 is an (m x d) matrix,
# b1 is a vector of length m, W2 is an (1 x m) row vector, 
# and b2 is a scalar. The total number of model parameters
# is md + m + m + 1 = md + 2m + 1.
#
# Define neural network architecture parameters
d = 5   # number of inputs for the airfoil dataset
m = 64  # number of neurons in the hidden layer

# Randomly initialize the neural network parameters to small values
W1 = np.sqrt((5/3)/d) * np.random.randn(d, m)
W2 = np.sqrt(1/m) * np.random.randn(m)
b1 = np.zeros(m)
b2 = 0
# Stack all neural network parameters into a single vector
theta_init = np.concatenate([W1.flatten(), b1, W2, [b2]])

print(f'Model initialized with {len(theta_init)} parameters')

# When calling minimize() we will specify maxnumfuneval = 1000 
# so that it will terminate when the total number of function evaluations is 1000.
maxnumfuneval = 1000  # maximum number of function evaluations

# We will also pass X_train, y_train, d, m to minimize() so that it can pass
# these variables to loss() which calculates the loss function and its gradients
theta_optimal, obj_history, grad_obj_history, nf = minimize(theta_init, loss, 
                                                            args=(X_train_norm, y_train, d, m),
                                                            maxnumlinesearch=None, 
                                                            maxnumfuneval=maxnumfuneval, red=1.0,
                                                            verbose=True)

# Generate predictions on training and test datasets to evaluate model performance
print('\nEvaluating model performance...')
y_train_pred = model_predict(theta_optimal, X_train_norm, d, m)
y_test_pred = model_predict(theta_optimal, X_test_norm, d, m)

# Compute mean square error on training and test datasets
train_mse = np.mean((y_train - y_train_pred)**2)
test_mse = np.mean((y_test - y_test_pred)**2)

# Compute R^2 values (close to 1 means the model is performing well)
train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

# Display results
print('\nPerformance Metrics:')
print('%-10s %-15s %-15s' % ('Dataset', 'MSE', 'R^2'))
print('----------------------------------------')
print('%-10s %-15g %-15g' % ('Training', train_mse, train_r2))
print('%-10s %-15g %-15g' % ('Test', test_mse, test_r2))

# Plot 1: Objective function value vs. total number of function evaluations
plt.figure(figsize=(10, 6))
plt.semilogy(nf, obj_history, 'b-', linewidth=2)
plt.xlabel('Number of function evaluations')
plt.ylabel('Mean Squared Error')
plt.title('Training loss')
plt.grid(True)
plt.show()

# Plot 2: Plot l2 norm of the gradient vs total number function evaluations
plt.figure(figsize=(10, 6))
plt.semilogy(nf, grad_obj_history, 'b-', linewidth=2)
plt.xlabel('Number of function evaluations')
plt.ylabel('l2 norm of gradient')
plt.title('Gradient norm')
plt.grid(True)
plt.show()

# Plot 3: Output predicted by model vs the actual value for test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, marker='.')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Set: Predicted vs Actual')
plt.grid(True)
plt.show()