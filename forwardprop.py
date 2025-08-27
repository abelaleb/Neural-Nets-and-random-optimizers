# forwardprop.py
import numpy as np
from backprop import backward_propagation

# ---------- Activations ----------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # stable softmax
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ---------- Parameters ----------
def initialize_parameters(input_size, h1, h2, h3, output_size, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((input_size, h1)) * np.sqrt(2 / input_size)
    b1 = np.zeros((1, h1))
    W2 = rng.standard_normal((h1, h2)) * np.sqrt(2 / h1)
    b2 = np.zeros((1, h2))
    W3 = rng.standard_normal((h2, h3)) * np.sqrt(2 / h2)
    b3 = np.zeros((1, h3))
    W4 = rng.standard_normal((h3, output_size)) * np.sqrt(2 / h3)
    b4 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3, W4, b4

# ---------- Forward ----------
def forward_propagation(
    X, W1, b1, W2, b2, W3, b3, W4, b4,
    keep_probs=(1.0, 1.0, 1.0), training=False, rng=None
):
    """
    Inverted Dropout:
    - During training: after ReLU, mask activations with Bernoulli(keep_prob) and divide by keep_prob
      so that expected value stays the same.
    - During inference: keep_probs are ignored (treated as 1.0).
    Returns:
        (Z1, A1, Z2, A2, Z3, A3, Z4, A4), dropout_masks
        dropout_masks: tuple(M1, M2, M3) used on A1, A2, A3 (or Nones if not training)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Layer 1
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    M1 = None
    if training and keep_probs[0] < 1.0:
        M1 = (rng.random(A1.shape) < keep_probs[0]).astype(A1.dtype)
        A1 = (A1 * M1) / keep_probs[0]

    # Layer 2
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    M2 = None
    if training and keep_probs[1] < 1.0:
        M2 = (rng.random(A2.shape) < keep_probs[1]).astype(A2.dtype)
        A2 = (A2 * M2) / keep_probs[1]

    # Layer 3
    Z3 = A2 @ W3 + b3
    A3 = relu(Z3)

    M3 = None
    if training and keep_probs[2] < 1.0:
        M3 = (rng.random(A3.shape) < keep_probs[2]).astype(A3.dtype)
        A3 = (A3 * M3) / keep_probs[2]

    # Output
    Z4 = A3 @ W4 + b4
    A4 = softmax(Z4)

    return (Z1, A1, Z2, A2, Z3, A3, Z4, A4), (M1, M2, M3)

# ---------- Training ----------
def train(
    X_train, y_train,
    W1, b1, W2, b2, W3, b3, W4, b4,
    epochs=300, learning_rate=0.05, lambda_reg=1e-3,
    keep_probs=(1.0, 1.0, 1.0), print_every=10,
    X_val=None, y_val=None, seed=123
):
    rng = np.random.default_rng(seed)
    history = {"loss": [], "acc": []}
    if X_val is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    for epoch in range(1, epochs + 1):
        # forward (training=True to apply dropout)
        (Z1, A1, Z2, A2, Z3, A3, Z4, A4), masks = forward_propagation(
            X_train, W1, b1, W2, b2, W3, b3, W4, b4,
            keep_probs=keep_probs, training=True, rng=rng
        )

        # backward
        W1, b1, W2, b2, W3, b3, W4, b4, loss = backward_propagation(
            X_train, y_train,
            Z1, A1, Z2, A2, Z3, A3, Z4, A4,
            W1, b1, W2, b2, W3, b3, W4, b4,
            learning_rate, lambda_reg,
            dropout_masks=masks
        )

        # metrics: train
        y_pred_train = np.argmax(A4, axis=1)
        acc = np.mean(y_pred_train == y_train)
        history["loss"].append(loss)
        history["acc"].append(acc)

        # metrics: val
        if X_val is not None and y_val is not None:
            # do a clean forward pass (no dropout)
            (_, _, _, _, _, _, _, A4_val), _ = forward_propagation(
                X_val, W1, b1, W2, b2, W3, b3, W4, b4,
                keep_probs=(1.0, 1.0, 1.0), training=False
            )
            eps = 1e-15
            ce_val = -np.sum(np.log(A4_val[np.arange(X_val.shape[0]), y_val] + eps)) / X_val.shape[0]
            reg_val = (lambda_reg / (2 * X_val.shape[0])) * (
                np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2)
            )
            val_loss = ce_val + reg_val
            val_acc = np.mean(np.argmax(A4_val, axis=1) == y_val)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if epoch % print_every == 0:
            if X_val is None:
                print(f"Epoch {epoch:4d} | loss={loss:.4f} | acc={acc*100:.2f}%")
            else:
                print(
                    f"Epoch {epoch:4d} | loss={loss:.4f} acc={acc*100:.2f}% "
                    f"| val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]*100:.2f}%"
                )

    return history, (W1, b1, W2, b2, W3, b3, W4, b4)

# ---------- Inference helpers ----------
def predict_proba(X, W1, b1, W2, b2, W3, b3, W4, b4):
    (_, _, _, _, _, _, _, A4), _ = forward_propagation(
        X, W1, b1, W2, b2, W3, b3, W4, b4, keep_probs=(1.0, 1.0, 1.0), training=False
    )
    return A4

def predict(X, W1, b1, W2, b2, W3, b3, W4, b4):
    return np.argmax(predict_proba(X, W1, b1, W2, b2, W3, b3, W4, b4), axis=1)
