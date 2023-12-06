import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from vae import VAE
from dataset import BioData
from utils import visualize_sample, visualize_comparison
from tqdm import tqdm
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    # 'batch' is a list of tensors
    # Concatenate tensors along the first dimension (dim=0)
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

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

# dataset
batch_size = 2 # batch 1 = onne scan file
bio_dataset = BioData()
train_size = int(0.99 * len(bio_dataset))
test_size = len(bio_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(bio_dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
print("Total number of scan files considered: ", len(bio_dataset))
# visualize_sample(bio_dataset[214], bio_dataset.min_val, bio_dataset.max_val)

# train the VAE
model = VAE(input_dim=442, hidden_dim=221, latent_dim=110, device=device).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs=100, device=device, train_loader=train_loader)

# test and save the model
average_loss = test(model, device, test_loader)
torch.save(model.state_dict(), f'models/average_test_loss_{average_loss}.pth')


# plot examples for validation
num_samples_to_visualize = 5
min_val = bio_dataset.min_val
max_val = bio_dataset.max_val
model.eval()

# setting the testloader to batch size 1
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

with torch.no_grad():
    for i, x in enumerate(test_loader):
        if i >= num_samples_to_visualize:
            break
        x = x.to(device)
        reconstructed_x, _, _ = model(x)
        
        # Reshape tensors if necessary to match the expected 1D format
        x_flat = x.view(-1)
        reconstructed_x_flat = reconstructed_x.view(-1)

        visualize_comparison(x_flat, reconstructed_x_flat, min_val, max_val, i)
