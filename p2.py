import numpy as np

class Model:
    def __init__(self, epochs = 10, lr = 0.1):
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.lr = lr
    def predict(self, x):
        y = np.dot(self.weights, x) + self.bias
        return 1 if y > 0 else 0
    def train(self, x, y):
        _, n_features = x.shape
        self.bias = 0
        self.weights = np.zeros(n_features)
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(x, y):
                pred = self.predict(xi)
                err = yi - pred
                self.weights += self.lr * err * xi
                self.bias += self.lr * err
                if err != 0:
                    errors += 1
            if errors == 0:
                break
            