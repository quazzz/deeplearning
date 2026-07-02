import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

samples = 50
x = np.linspace(0, 10, samples)
y = 2*x + 5+ np.random.normal(0, 1, samples)
outlier_indices = [10, 25, 40]

y[outlier_indices] += np.array([40, -35, 45])

#plt.scatter(x,y)
#plt.scatter(x[outlier_indices], y[outlier_indices], c='red')
#plt.show()

def train(x, y, losstype='MSE', lr =0.01, epochs = 1000):
    w,b = 0.0, 0.0
    n = len(x)
    losses = []
    for _ in range(epochs):
        y_pred = x * w + b
        err = y_pred - y
        if losstype=='MSE':
            loss = np.mean(err ** 2)
            dw = (2 / n) * np.sum(err * x)
            db = (2 / n) * np.sum(err)
        elif losstype=='MAE':
            loss = np.mean(np.abs(err))
            dw = (1 / n) * np.sum(np.sign(err) * x)
            db = (1 / n) * np.sum(np.sign(err))
        w -= dw * lr
        b -= db * lr
        losses.append(loss)
    return w,b, losses
w_mse, b_mse, loss_mse = train(x,y,losstype='MSE')
w_mae, b_mae, loss_mae = train(x,y, losstype='MAE')
print(f'y={w_mse:.2}x + {b_mse:.2}')
print(f'y={w_mae:.2}x + {b_mae:.2}')

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='data', alpha=0.6)
plt.scatter(x[outlier_indices], y[outlier_indices], color='red', label='outliers', zorder=5)

x_line = np.linspace(0, 10, 100)
plt.plot(x_line, w_mse * x_line + b_mse, 'g-', linewidth=2, label=f'MSE fit (w={w_mse:.2f})')
plt.plot(x_line, w_mae * x_line + b_mae, 'orange', linewidth=2, label=f'MAE fit (w={w_mae:.2f})')
plt.plot(x_line, 2 * x_line + 5, 'k--', alpha=0.5, label='true relationship')

plt.legend()
plt.title("MSE vs MAE: Outlier Sensitivity")
plt.xlabel('x')
plt.ylabel('y')
plt.show()