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
from utils import visualize_sample, visualize_comparison
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### dataset
dir_path = '/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data'
bio_dataset = BioData(dir_path)
print("number of samples in dataset: ", len(bio_dataset))
print("shape of first sample: ", bio_dataset[0].shape)
# visualize_sample(bio_dataset[214], bio_dataset.min_val, bio_dataset.max_val)

# split dataset into training and test set (80/20)
train_size = int(0.8 * len(bio_dataset))
test_size = len(bio_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(bio_dataset, [train_size, test_size])

## create dataloader for training
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
print("batch size: ", batch_size)
print("number of batches in train loader: ", len(train_loader))
print("number of batches in test loader: ", len(test_loader))


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


### test the VAE
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # No gradients needed for testing
        for x in test_loader:
            x = x.to(device)

            # Forward pass through the model
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            test_loss += loss.item()

    # Compute the average loss
    average_loss = test_loss / len(test_loader.dataset)
    print("\tTest Set Average Loss: ", average_loss)
    return average_loss

average_loss = test(model, device, test_loader)

torch.save(model.state_dict(), f'models/average_test_loss_{average_loss}.pth')


### plot examples

num_samples_to_visualize = 5
min_val = bio_dataset.min_val
max_val = bio_dataset.max_val
model.eval()
with torch.no_grad():
    # Assuming test_loader is available and has batch_size of 1 for simplicity
    for i, x in enumerate(test_loader):
        if i >= num_samples_to_visualize:
            break
        x = x.to(device)
        reconstructed_x, _, _ = model(x)
        
        # Reshape tensors if necessary to match the expected 1D format
        x_flat = x.view(-1)
        reconstructed_x_flat = reconstructed_x.view(-1)

        visualize_comparison(x_flat, reconstructed_x_flat, min_val, max_val)

