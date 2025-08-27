# backprop.py (updated)
import numpy as np

def backward_propagation(
    X, y,
    Z1, A1, Z2, A2, Z3, A3, Z4, A4,
    W1, b1, W2, b2, W3, b3, W4, b4,
    lambda_reg,
    dropout_masks=None
):
    """
    Returns gradients only. Does NOT update parameters.
    Used with Adam or other optimizers.
    """
    m = X.shape[0]

    # ----- Loss (cross-entropy + L2) -----
    eps = 1e-15
    ce = -np.sum(np.log(A4[np.arange(m), y] + eps)) / m
    reg = (lambda_reg / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    loss = ce + reg

    # ----- Output layer -----
    dZ4 = A4.copy()
    dZ4[np.arange(m), y] -= 1
    dW4 = (A3.T @ dZ4) / m + (lambda_reg / m) * W4
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    # ----- Hidden 3 -----
    dA3 = dZ4 @ W4.T
    if dropout_masks is not None and dropout_masks[2] is not None:
        dA3 *= dropout_masks[2]  # Inverted dropout: scale at train time
    dZ3 = dA3 * (Z3 > 0)
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

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, loss