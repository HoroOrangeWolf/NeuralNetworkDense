import torch.utils.data
import torchvision.datasets

from NeuralNetwork import MNISTNeuralNetwork, trainNeuralNetwork, testNeuralNetwork

batch_size = 32
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST("/data", train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST("/data", train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = MNISTNeuralNetwork(1)
model.to(device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for x in range(epochs):
    print(f'Epochs: {x} \n --------------------------------------------------------')
    trainNeuralNetwork(data_loader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
    testNeuralNetwork(data_loader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
