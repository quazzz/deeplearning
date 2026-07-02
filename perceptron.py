import numpy as np
import matplotlib.pyplot as plt

# our perceptron model 

class Perceptron:
    def __init__(self, lr=0.1, epochs = 10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def predict(self, x):
        output = np.dot(x, self.weights) + self.bias # y = ax + b
        return 1 if output > 0 else 0
    def train(self, x, y):
        _, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(x, y):
                pred = self.predict(xi)
                error = yi - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr*error
                if error != 0:
                    errors += 1
            print(f'epoch: {epoch}, errors: {errors}, weights: {self.weights}, bias: {self.bias}')
            if errors == 0:
                break     

# AND and OR gate labels
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0,1,1,1])

fig, axs = plt.subplots(1, 2)

prtron = Perceptron(lr=0.05, epochs= 20)
prtron.train(X,y_and)
preds = np.array([prtron.predict(xi) for xi in X])

for xi, yi in zip(X, y_and):
    pred = prtron.predict(xi)
    print(f'pred: {pred}, actual: {yi}')

axs[0].scatter(X[:, 0], X[:, 1])
w1, w2 = prtron.weights
b = prtron.bias

x_vals = np.linspace(-0.5, 1.5, 100)
if w2 != 0:
    y_vals = -(w1 * x_vals + b) / w2 # from y = w1*x1 + w2*x2 + b (our perceptron main formula)
    axs[0].plot(x_vals, y_vals)


prtron2 = Perceptron(lr = 0.05, epochs = 20)
prtron2.train(X, y_or)

axs[1].scatter(X[:, 0], X[:, 1])
w1, w2 = prtron2.weights
b = prtron2.bias
x_vals = np.linspace(-0.5, 1.5, 100)

if w2 != 0:
    y_vals = -(w1 * x_vals + b) / w2
    axs[1].plot(x_vals, y_vals)

for xi, yi in zip(X, y_or):
    pred = prtron2.predict(xi)
    print(f'pred: {pred}, actual: {yi}')
plt.show()