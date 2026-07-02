import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data = datasets.MNIST(root='./data', train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
def onehot(labels, classes = 10):
    return torch.eye(classes)[labels]
def train(type, epochs=10):
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        totloss = 0
        corr = 0
        tot = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            if type=='MSE':
                probs = torch.softmax(logits, dim=1)
                targets = onehot(labels)
                loss = nn.functional.mse_loss(probs, targets)
            elif type =='CE':
                loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            totloss += loss.item()
            preds = torch.argmax(logits, dim = 1)
            corr += (preds == labels).sum().item()
            tot += labels.size(0)
        avg_loss = totloss / len(train_loader)
        accuracy = corr / tot
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
    return train_losses, train_accuracies

torch.manual_seed(42)
mse_losses, mse_accuracies = train('MSE')
torch.manual_seed(42)
ce_losses, ce_accuracies = train('CE')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(mse_losses, label='MSE')
axes[0].plot(ce_losses, label='Cross-Entropy')
axes[0].set_title('Training Loss (not directly comparable in scale)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(mse_accuracies, label='MSE')
axes[1].plot(ce_accuracies, label='Cross-Entropy')
axes[1].set_title('Training Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()
