import numpy as np

def loss(theta, X, y, d, m):
    """
    Compute loss function and its gradients wrt the parameters
    of a feedforward neural network with single hidden layer
    for approximating a scalar function of d variables, i.e.,
    y(x) = W2*tanh(W1*x + b1) + b2
    where x is a vector of length d, W1 is an (m x d) matrix,
    b1 is a vector of length m, W2 is an (1 x m) matrix, and
    b2 is a scalar. The total number of model parameters
    is md + m + m + 1 = md + 2m + 1. We use a tanh()
    activation function - if you want to train a model with
    a different activation function, edit lines 60 and 82.
    
    This function is vectorized to compute the loss function
    and its gradients for a batch of n training points.
    
    model_predict.py is the function used for making predictions
    after the model parameters have been estimated.
    
    Inputs:
    theta - Vector containing all the neural network parameters
            The first d*m entries correspond to W1
            The next m entries correspond to b1
            The next m entries correspond to W2
            The last scalar is b2
    X - N x d matrix of input features,
        where N is the number of points in the batch
    y - n x 1 vector of regression target values for the points in the batch
    d - number of neural network inputs
    m - number of neurons in the hidden layer
    
    Outputs:
    loss - mean squared error loss
    grad - n x 1 vector containing gradient of loss with respect to theta,
           where n = md + 2m + 1 is number of parameters for a neural 
           network with single hidden layer when approximating a scalar function 
           of d variables
    """
    # Number of samples
    N, _ = X.shape
    
    # Extract neural network parameters from theta vector
    # Following PyTorch convention for shapes of weight matrices
    idx = 0
    # Extract W1 matrix of size (5 x 64)
    W1 = np.reshape(theta[idx:idx+d*m], (d, m))
    idx = idx + d*m
    # Extract b1 vector of length (64 x 1)
    b1 = theta[idx:idx+m]
    idx = idx + m
    # Extract W2 vector of length (64 x 1)
    W2 = theta[idx:idx+m]
    idx = idx + m
    # Extract b2 (scalar)
    b2 = theta[idx]
    
    # Generate model predictions for all points in X
    Z1 = np.dot(X, W1) + np.tile(b1.reshape(1, -1), (N, 1))  # N x 64
    A1 = np.tanh(Z1)  # N x 64
    y_pred = np.dot(A1, W2) + b2  # N x 1
    
    # Compute loss (mean squared error)
    loss_value = np.mean((y - y_pred)**2)
    
    # Compute gradients of loss wrt to model parameters
    # using backpropagation. First initialize gradients to 0
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = 0
    
    # Output layer gradients
    dL_dy_pred = -2 * (y - y_pred) / N  # N x 1
    
    # Gradient for output layer
    db2 = np.sum(dL_dy_pred)
    dW2 = np.dot(A1.T, dL_dy_pred)
    
    # Gradient for hidden layer
    dL_dA1 = np.outer(dL_dy_pred, W2)  # N x 64
    dL_dZ1 = dL_dA1 * (1 - A1**2)  # N x 64, derivative of tanh is (1 - tanh^2)
    
    # Gradients for W1 and b1
    db1 = np.sum(dL_dZ1, axis=0)
    dW1 = np.dot(X.T, dL_dZ1)
    
    # Assemble gradients into a single vector in the same order as theta
    grad = np.zeros_like(theta)
    idx = 0
    
    # Stack gradients of loss wrt W1
    grad[idx:idx+d*m] = dW1.flatten()
    idx = idx + d*m
    
    # Stack gradients of loss wrt b1
    grad[idx:idx+m] = db1
    idx = idx + m
    
    # Stack gradients of loss wrt W2
    grad[idx:idx+m] = dW2
    idx = idx + m
    
    # Stack gradients of loss wrt b2
    grad[idx] = db2
    
    return loss_value, grad
