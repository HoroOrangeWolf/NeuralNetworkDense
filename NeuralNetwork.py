import torch
from torch import nn


class MNISTNeuralNetwork(nn.Module):
    def __init__(self, in_channels):
        super(MNISTNeuralNetwork, self).__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=22500, out_features=1028),
            nn.ReLU(),
            nn.Linear(in_features=1028, out_features=1028),
            nn.ReLU(),
            nn.Linear(in_features=1028, out_features=10)
        )

    def forward(self, x):
        return self.sequence(x)


def trainNeuralNetwork(data_loader, model, optimizer, loss_fn, device):
    size = len(data_loader.dataset)
    avg_loss = 0
    for count, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        if count % 100 == 0 and count != 0:
            print(f'avg. loss: [{avg_loss/(100*len(x)):>5f}  [{count*len(x)}/{size}]')
            avg_loss = 0


def testNeuralNetwork(data_loader, model, loss_fn, device):
    size = len(data_loader.dataset)
    avg_loss, correct = 0, 0

    with torch.no_grad():
        for count, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        print(f'avg.loss: {(avg_loss / size):>4f} correct: {(correct / size)*100:>2f}%')




