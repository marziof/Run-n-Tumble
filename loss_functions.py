import numpy as np


def loss_diff(position, new_position, loss_function,  X, y):
    """Calculate the difference in loss"""
    return loss_function(new_position, X, y) - loss_function(position, X, y)

def MSE_loss(position, X, y):
    """Calculate the squared loss ||position - target||^2""" 
    # position = weights; X = samples; y = labels
    predictions = X @ position  # Model predictions
    return np.mean((predictions - y) ** 2)  # Sum of squared errors


def huber_loss(position, X, y, delta=1.0, l2_lambda=0.01):
    """
    Computes the Huber loss + L2 regularization for stability.
    
    Parameters:
    - position: Current weight vector (d,)
    - X: Feature matrix (n, d)
    - y: Target values (n,)
    - delta: Threshold for switching between squared loss and absolute loss
    - l2_lambda: Regularization strength for weight penalty

    Returns:
    - loss: Computed loss value (scalar)
    """
    predictions = X @ position  # Model predictions
    residuals = predictions - y  # Errors

    # Huber loss formula
    is_small_error = np.abs(residuals) <= delta
    squared_loss = 0.5 * residuals**2
    linear_loss = delta * (np.abs(residuals) - 0.5 * delta)
    
    huber = np.where(is_small_error, squared_loss, linear_loss)
    
    # L2 regularization (prevents overfitting)
    l2_penalty = l2_lambda * np.sum(position**2)
    
    return np.mean(huber) + l2_penalty