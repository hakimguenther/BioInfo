import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from vae import VAE
from dataset import BioData
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### dataset
dir_path = '/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data'
bio_dataset = BioData(dir_path)
print("number of samples in dataset: ", len(bio_dataset))
print("shape of first sample: ", bio_dataset[0].shape)


## create dataloader for training
batch_size = 100
train_loader = DataLoader(dataset=bio_dataset, batch_size=batch_size, shuffle=True)


### train the VAE
model = VAE(input_dim=442, hidden_dim=221, latent_dim=110, device=device).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device, train_loader):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / len(train_loader.dataset))
    return overall_loss

train(model, optimizer, epochs=1, device=device, train_loader=train_loader)

torch.save(model.state_dict(), f'model.pth')

