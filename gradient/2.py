import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

x = np.linspace(0.5, 5, 50)
y = x*3 + 2 + np.random.normal(0, 0.5 ,50)
X=torch.tensor(x, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

def train(name, epochs = 300, lr = 0.01):
    torch.manual_seed(0)
    model = Net()
    criterion = nn.MSELoss()
    if name =='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        err = criterion(pred, Y)
        err.backward()
        optimizer.step()
        losses.append(err.item())
    return losses
sgd_losses = train('sgd')
adam_losses = train('adam')

import matplotlib.pyplot as plt
plt.plot(sgd_losses, label='sgd')
plt.plot(adam_losses, label='adam')
plt.legend()
plt.show( )

lrs = np.logspace(-5, 1, 10)
final_losses = []

for lr in lrs:
    losses = train('sgd', lr=lr, epochs=300)
    final_loss = losses[-1]
    final_losses.append(final_loss)

plt.figure(figsize=(8,5))
plt.plot(lrs, final_losses, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Learning rate (log scale)")
plt.ylabel("Final loss (log scale)")
plt.title("Learning Rate Finder")
plt.show()