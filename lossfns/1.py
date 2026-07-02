import numpy as np

def MSE(y, pred):
    return np.mean((y - pred) ** 2) 
def MAE(y, pred):
    return np.mean(np.abs(y - pred))
def LOGLOSS(y, pred):
    return -np.mean((y * np.log(pred)) + (1 - y) * np.log(1 - pred))
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    return loss

# 3 samples, 3 classes, one-hot encoded
y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

y_pred = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6]
])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"Loss: {loss:.4f}")