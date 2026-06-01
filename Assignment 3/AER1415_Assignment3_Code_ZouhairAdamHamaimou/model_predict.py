import numpy as np

def model_predict(theta, X, d, m):
    """
    Prediction function for a feedforward neural network
    with a single hidden layer for approximating a scalar
    function of d variables, i.e.,
    y(x) = W2*tanh(W1*x + b1) + b2
    where x is a vector of length d, W1 is an (m x d) matrix,
    b1 is a vector of length m, W2 is an (1 x m) row vector,
    and b2 is a scalar. The total number of model parameters
    is md + m + m + 1 = md + 2m + 1. We assume a tanh()
    activation; see line 57.
    
    This function is vectorized to calculate model predictions
    for a batch of n inputs.
    
    loss.py provides a function to calculate the mean-square error
    for a batch of training points and compute its first-order
    derivatives wrt the model parameters.
    
    Inputs:
    theta - Vector containing all the model parameters
            The first d*m entries correspond to W1
            The next m entries correspond to b1
            The next m entries correspond to W2
            The last scalar is b2
    X - N x d matrix of input features, where N is the number of points
        at which model predictions are sought
    d - number of neural network inputs
    m - number of neurons in the hidden layer
    
    Output:
    y_pred - N x 1 vector of predicted values
    """
    # Extract parameters from theta vector
    idx = 0
    # Following PyTorch convention for shapes of the weight matrices
    # W1 \in R^{5x64}, W2 \in \R^{64x1}
    # Extract W1 matrix of size (5 x 64)
    W1 = np.reshape(theta[idx:idx+d*m], (d, m))
    idx = idx + d*m
    # Extract b1 vetor of length (64 x 1)
    b1 = theta[idx:idx+m]
    idx = idx + m
    # Extract W2 vector of length (64 x 1)
    W2 = theta[idx:idx+m]
    idx = idx + m
    # Extract b2 (scalar)
    b2 = theta[idx]
    
    # Ensure X is properly oriented
    N, _ = X.shape
    
    # Predict model output for all N points in X
    Z1 = np.dot(X, W1) + np.tile(b1.reshape(1, -1), (N, 1))  # N x 64
    A1 = np.tanh(Z1)  # N x 64
    y_pred = np.dot(A1, W2) + b2  # N x 1
    
    return y_pred
