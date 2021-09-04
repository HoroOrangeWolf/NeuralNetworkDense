import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.normalization = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

    def forward(self, x):
        bn = self.normalization(x)

        conv1 = self.relu(self.conv1(bn))

        conv2 = self.relu(self.conv2(conv1))

        conv2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(conv2_dense))

        conv3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(conv3_dense))

        conv4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(conv4_dense))

        conv5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return conv5_dense


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class MNISTNeuralNetwork(nn.Module):
    def __init__(self, in_channels):
        super(MNISTNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, stride=(1, 1), kernel_size=(2, 2), bias=False)

        self.relu = nn.ReLU()

        self.dense1 = self.make_dense_block(DenseBlock, in_channels=64)

        self.dense2 = self.make_dense_block(DenseBlock, in_channels=128)

        self.dense3 = self.make_dense_block(DenseBlock, in_channels=128)

        self.trans1 = self.make_transition_layer(TransitionLayer, in_channels=160, out_channels=128)

        self.trans2 = self.make_transition_layer(TransitionLayer, in_channels=160, out_channels=128)

        self.trans3 = self.make_transition_layer(TransitionLayer, in_channels=160, out_channels=64)

        self.bn = nn.BatchNorm2d(num_features=64)

        self.pre_classifier = nn.Linear(in_features=576, out_features=512)

        self.classifier = nn.Linear(in_features=512, out_features=10)

    def make_dense_block(self, block, in_channels):
        layers = [block(in_channels)]
        return nn.Sequential(*layers)

    def make_transition_layer(self, layer, in_channels, out_channels):
        modules = [layer(in_channels, out_channels)]
        return nn.Sequential(*modules)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        out = self.dense1(conv1)
        out = self.trans1(out)

        out = self.dense2(out)
        out = self.trans2(out)

        out = self.dense3(out)
        out = self.trans3(out)

        u = nn.Flatten()

        out = u(out)

        out = self.pre_classifier(out)

        out = self.relu(out)

        out = self.classifier(out)

        return out


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
            print(f'avg. loss: [{avg_loss / (100 * len(x)):>5f}  [{count * len(x)}/{size}]')
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

        print(f'avg.loss: {(avg_loss / size):>4f} correct: {(correct / size) * 100:>2f}%')
