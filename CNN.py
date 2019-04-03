
# Import modules

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time

# Run on GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Get data

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4), # Crop of the image at a random location, of size 32*32, with padding on the 4 corners
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

# Models

# Multilayer perceptron (MLP)
class ModelMLP(nn.Module):
    def __init__(self):
        super(ModelMLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 3072)
        self.fc2 = nn.Linear(3072, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 320)
        self.fc5 = nn.Linear(320, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# Convolutional Neural Network (CNN)
class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Test the model

def test_model(model):
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # Put data on device used
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

# Train the model

def train_model(model, criterion, optimizer, num_epochs):
    
    loss_epoch = []
    acc_epoch = []

    for epoch in range(num_epochs):
    
        total_loss = 0
        
        t0 = time.time()
    
        for i, (images, labels) in enumerate(train_loader):
            
            # Put data on device used
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
        
        # Save results every epoch
        loss_epoch.append(total_loss / i)
        acc = test_model(model)
        acc_epoch.append(acc)
        
        # Print progress
        if epoch%10 == 0:
            print()
        print(round(time.time() - t0), end="|")
        
    print()
    
    return loss_epoch, acc_epoch

# Main

import matplotlib.pyplot as plt

# Parameters
# Models = {"MLP": ModelMLP}
Models = {"CNN": ModelCNN}
num_epochs = 100
criterion = nn.CrossEntropyLoss()
lrs = [0.01]
dfs = [0.9]
weight_decay = 8e-4

for model_name, Model in Models.items():
    
    for df in dfs:

        losses = {}
        accs = {}
        for lr in lrs:

            model = Model()
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                model = nn.DataParallel(model)
            model.to(device)

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=df, weight_decay=weight_decay)
            loss_epoch, acc_epoch = train_model(model, criterion, optimizer, num_epochs)
            losses[lr] = loss_epoch
            accs[lr] = acc_epoch
            acc = test_model(model)

            print('Model: {}, epochs: {}'.format(model_name, num_epochs))
            print('Learning rate: {}, discount factor: {}, weight decay: {}'.format(lr, df, weight_decay))
            print()
            print('Loss at each epoch:')
            print(loss_epoch)
            print('Acc at each epoch:')
            print(acc_epoch)
            print()
            print('Accuracy of the model on the test images: %f %%' % acc)
            print()

        args = [z for y in losses.values() for z in [list(range(1, len(y) + 1)), y]]
        plt.plot(*args)
        plt.suptitle('Model: {}'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Average loss')
        plt.legend(losses.keys(), title='Learning rate')
        plt.savefig('{}-loss.pdf'.format(model_name))
        plt.show()
        
        args = [z for y in accs.values() for z in [list(range(1, len(y) + 1)), y]]
        plt.plot(*args)
        plt.suptitle('Model: {}'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Testing accuracy')
        plt.legend(accs.keys(), title='Learning rate')
        plt.savefig('{}-acc.pdf'.format(model_name))
        plt.show()
