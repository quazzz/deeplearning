import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
x = np.linspace(0.5, 5, 50)
y = x*3 + 2 + np.random.normal(0, 0.5 ,50)
w,b =0.0, 0.0
lr = 0.01
epochs = 1000
n = len(x)
losses = []
for _ in range(epochs):
    pred = w*x+b
    err = pred - y

    loss = np.mean(err ** 2) # MSE
    losses.append(loss)

    dw = (2 / n) * np.sum(err * x)
    db = (2 / n) * np.sum(err)

    w -= lr * dw
    b -= lr * db


plt.plot(losses)
plt.show()