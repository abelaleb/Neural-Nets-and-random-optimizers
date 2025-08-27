# backprop.py
import numpy as np

"""
Vanilla Batch Gradient Descent with:
- Cross-entropy loss
- L2 regularization
- Inverted Dropout (applied in forward; here we mask gradients)

Update rule per epoch (full batch):
    param := param - learning_rate * grad
"""

def backward_propagation(
    X, y,
    Z1, A1, Z2, A2, Z3, A3, Z4, A4,
    W1, b1, W2, b2, W3, b3, W4, b4,
    learning_rate, lambda_reg,
    dropout_masks=None
):
    """
    dropout_masks: tuple of masks (M1, M2, M3) used on A1, A2, A3 during forward pass,
                   or None if dropout disabled. Inverted dropout is used, so simply
                   multiply dA by the same mask.
    """
    m = X.shape[0]

    # ----- Loss (cross-entropy + L2) -----
    # Small epsilon to avoid log(0)
    eps = 1e-15
    ce = -np.sum(np.log(A4[np.arange(m), y] + eps)) / m
    reg = (lambda_reg / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    loss = ce + reg

    # ----- Output layer -----
    dZ4 = A4.copy()
    dZ4[np.arange(m), y] -= 1                 # dL/dZ4
    dW4 = (A3.T @ dZ4) / m + (lambda_reg / m) * W4
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    # ----- Hidden 3 -----
    dA3 = dZ4 @ W4.T                          # dL/dA3
    # apply dropout mask (inverted dropout)
    if dropout_masks is not None and dropout_masks[2] is not None:
        dA3 *= dropout_masks[2]
    dZ3 = dA3 * (Z3 > 0)                      # ReLU'
    dW3 = (A2.T @ dZ3) / m + (lambda_reg / m) * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # ----- Hidden 2 -----
    dA2 = dZ3 @ W3.T
    if dropout_masks is not None and dropout_masks[1] is not None:
        dA2 *= dropout_masks[1]
    dZ2 = dA2 * (Z2 > 0)
    dW2 = (A1.T @ dZ2) / m + (lambda_reg / m) * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # ----- Hidden 1 -----
    dA1 = dZ2 @ W2.T
    if dropout_masks is not None and dropout_masks[0] is not None:
        dA1 *= dropout_masks[0]
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (X.T @ dZ1) / m + (lambda_reg / m) * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # ----- Update -----
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    return W1, b1, W2, b2, W3, b3, W4, b4, loss
