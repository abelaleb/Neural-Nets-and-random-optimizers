# train.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from forwardprop import (
    forward_propagation,
    initialize_parameters,
    train,
    predict_proba
)

# 1) Load and prepare data  (Iris: 150 samples, 4 features, 3 classes)
iris = load_iris()
X = iris.data.astype(np.float64)   # shape: (150, 4)
y = iris.target.astype(int)        # labels: {0,1,2}

print(f"feature shape: {X.shape}")
print(f"target shape: {y.shape}")

# 2) Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Define network sizes (3 hidden layers)
input_size  = X.shape[1]   # 4 features
hidden1     = 32
hidden2     = 16
hidden3     = 8
output_size = 3            # 3 classes

# 5) Initialize parameters (He init)
W1, b1, W2, b2, W3, b3, W4, b4 = initialize_parameters(
    input_size, hidden1, hidden2, hidden3, output_size, seed=42
)

# 6) Train
history, (W1, b1, W2, b2, W3, b3, W4, b4) = train(
    X_train, y_train,
    W1, b1, W2, b2, W3, b3, W4, b4,
    epochs=300,
    learning_rate=0.001,           # ← Adam default
    lambda_reg=1e-3,
    keep_probs=(0.9, 0.9, 0.9),
    print_every=20,
    X_val=X_test, y_val=y_test
)
# 7) Evaluate on test
from forwardprop import predict
y_pred = predict(X_test, W1, b1, W2, b2, W3, b3, W4, b4)
test_acc = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# 8) Plot training curves
plt.figure()
plt.plot(history["loss"], label="Train Loss")
if "val_loss" in history:
    plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (CE + L2)")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history["acc"], label="Train Acc")
if "val_acc" in history:
    plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
