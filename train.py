import torch
import torchvision
import argparse
import sys
import os
from PIL import Image
import numpy as np
from pprint import pprint as pp
from torchvision.transforms import transforms
from Network import Network
import torch.nn as nn
import torch.optim as optim
import atexit
from test import test

def exit_handler():
    print('My application is ending!')

atexit.register(exit_handler)

# Create a one hot matrix of the class number i.e. 3 -> [0, 0, 1, .... 0]
def to_one_hot(batch_size, num_classes, num):

    output = torch.zeros(num.size()[0], num_classes)

    #print(num.size()[0])
    for i in range(num.size()[0]):
        try:
            output[i][num[i]-1] = 1
        except:
            print(output.shape)
    return output

def load_class_names():
    with open('class_names.txt', 'r') as f:
        lines = f.readlines()
    return lines

def train():

    # Define training hyperparameters
    epochs = 100
    learning_rate = 0.01
    log_interval = 10
    training_batch_size = 64
    testing_batch_size = 1000
    crop_size = 128

    net = Network()
    print(net)

    transform = transforms.Compose(
        [transforms.RandomResizedCrop(crop_size), transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # Load Training Data...
    training_data_path = 'cars_train'
    dataset = torchvision.datasets.ImageFolder('cars_train',transform=transform)

    size = len(dataset)
    print(size)
    split = [int(size * 0.8) + 1, int(size * 0.2)]
    print(split, sum(split))

    [training_dataset, testing_dataset] = torch.utils.data.random_split(dataset, split)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=testing_batch_size, shuffle=True, num_workers=2)

    # Training Parameters
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    class_names = load_class_names()  # based on folder name
    num_classes = len(class_names)
    losses = []

    for i in range(0, epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()  # zero the gradient buffers
            y = to_one_hot(training_batch_size, num_classes, target)
            output = net(data)  # forward through the net
            #print([type(output), type(y)])
            loss = loss_function(output, y.float())  # calculate the loss
            loss.backward()  # back prop
            losses.append(loss.item())
            optimizer.step()  # update

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        test(net, test_loader, i)

    try:
        saved_networks = os.listdir('savedNetworks')
        num_networks = len(saved_networks)

        torch.save(net.state_dict(), 'savedNetworks/network_' + str(num_networks + 1))
        print('Network saved successfully.')

    except:
        print("Unable to save state dictionary")

if __name__ == "__main__":
    train()
