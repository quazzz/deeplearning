import numpy as np
np.random.seed(42)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=float)
y= np.array([[0], [1], [1], [0]])

def init_weights():
    np.random.seed(0)
    w1 = np.random.randn(2,3) * 0.1
    b1 = np.zeros((1, 3))
    w2 = np.random.randn(3, 1) * 0.1
    b2 = np.zeros((1, 1))
    return w1, b1, w2 ,b2
def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu_grad(z):
    return (z > 0).astype(float)
def forward(x, w1, w2, b1, b2):
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    cache = (x, z1, a1, z2, a2)
    return a2, cache

def bce_loss(y, pred, eps = 1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def backward(cache, y, w2):
    x, z1,a1,z2,a2 = cache
    n = len(y)
    dz2 = (a2 - y) / n
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=  0)
    da1 = dz2 @ w2.T
    dz1 = da1 * relu_grad(z1)
    dw1 =  x.T @ dz1
    db1 = np.sum(dz1, axis = 0)
    return dw1, db1, dW2, db2
w1, b1, w2, b2 = init_weights()
a2, cache = forward(x, w1, w2 ,b1, b2)
loss = bce_loss(y, a2)
dw1, db1, dW2, db2 = backward(cache, y, w2)

print(f'loss: {loss:.4f}')
print(f'dw1 shape: {dw1}, dw2: {dW2}')