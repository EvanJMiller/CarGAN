import torch
import torchvision
import argparse
import sys
import os
from PIL import Image
import numpy as np
from pprint import pprint as pp
from torchvision.transforms import transforms
import torchvision.utils as vutils
from Network import Network
import torch.nn as nn
import torch.optim as optim
import atexit
from test import test
import matplotlib.pyplot as plt
from datetime import datetime

from Generator import Generator
from Discriminator import Discriminator

# Create the generator
generator = Generator(128)
discriminator = Discriminator(128)

def exit_handler():
    if(os.path.isdir('/results')):
        torch.save(generator.state_dict(), "results/generator_results.pkl")
        torch.save(discriminator.state_dict(), "results/discriminator_results.pkl")
    else:
        torch.save(generator.state_dict(), "generator_results.pkl")
        torch.save(discriminator.state_dict(), "discriminator_results.pkl")
    print("Results Saved...")

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
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train():

    # Define training hyperparameters
    epochs = 5
    learning_rate = 0.02
    log_interval = 10
    training_batch_size = 128
    testing_batch_size = 1000
    crop_size = 28

    channels = 3
    image_size = 64

    #Adam optimizer parameter
    beta1 = 0.5

    # number of gpus
    ngpu = 1

    num_epochs = 5

    # Decide which device we want to run on
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    generator = Generator(128)
    discriminator = Discriminator(128)

    #if (device.type == 'cuda') and (ngpu > 1):
    #   generator = nn.DataParallel(generator, list(range(ngpu)))
    #   discriminator = nn.DataParallel(discriminator, list(range(ngpu)))


    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load Training Data...
    training_data_path = 'cars_train'

    #NOTE: Comment either out for training on a specific dataset

    #dataset = torchvision.datasets.ImageFolder('cars_train',transform=transform)
    dataset = torchvision.datasets.MNIST('./', train=True, transform=transform, target_transform=None, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=training_batch_size, shuffle=True,
                                               num_workers=4)

    # Plot some training images
    real_batch = next(iter(data_loader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    #plt.show()

    size = len(dataset)

    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()

    # Create batch of latent vectors
    fixed_noise = torch.randn(64, 100, 1, 1)
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(num_epochs):

        for i, data in enumerate(data_loader, 0):

            ## Train with real data batch
            discriminator.zero_grad()

            real_cpu = data[0]
            b_size = real_cpu.size(0)
            l = list(data[1].size())[0]

            output_real = torch.ones(l)
            output_fake = torch.zeros(l)
            output = discriminator(data[0]).view(-1)

            d_real_image_error = criterion(output, output_real)

            # Calculate gradients for D in backward pass
            d_real_image_error.backward()
            D_x = output.mean().item()

            #noise vector
            noise = torch.randn(b_size,100, 1, 1)
            # Generate fake image batch with G
            fake = generator(noise)
            #label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            d_fake_image_error = criterion(output, output_fake)
            # Calculate the gradients for this batch
            d_fake_image_error.backward()
            D_G_z1 = output.mean().item()
            errD = d_real_image_error + d_fake_image_error
            # Update D
            D_optimizer.step()

            #Update the Generator
            generator.zero_grad()
            #label.fill_(label)
            output = discriminator(fake).view(-1)
            generator_err = criterion(output, output_real)
            # Calculate gradients for G
            generator_err.backward()
            D_G_z2 = output.mean().item()
            G_optimizer.step()

            # Save Losses for plotting later
            G_losses.append(generator_err.item())
            D_losses.append(errD.item())

            iters += 1
            print("iters: " + str(iters))

        stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        save_network(generator, discriminator, stamp, epoch)

def save_network(generator, discriminator, stamp, epoch):

    if not os.path.isdir('saved_networks'):
        os.mkdir('saved_networks')

    torch.save(generator.state_dict(), "saved_networks/generator_" + str(epoch) + "_" + stamp)
    torch.save(discriminator.state_dict(), "saved_networks/discriminator_" + str(epoch) + "_" + stamp)


if __name__ == "__main__":
    train()
