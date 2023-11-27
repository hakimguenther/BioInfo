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
from utils import visualize_sample
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### dataset
dir_path = '/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data'
bio_dataset = BioData(dir_path)
print("number of samples in dataset: ", len(bio_dataset))
print("shape of first sample: ", bio_dataset[0].shape)
# visualize_sample(bio_dataset[214], bio_dataset.min_val, bio_dataset.max_val)


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

from tqdm import tqdm

def train(model, optimizer, epochs, device, train_loader):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, x in progress_bar:

            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            # Update progress bar description with the current loss
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f}")

        average_loss = overall_loss / len(train_loader.dataset)
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", average_loss)
    return overall_loss

train(model, optimizer, epochs=1, device=device, train_loader=train_loader)

torch.save(model.state_dict(), f'model.pth')

