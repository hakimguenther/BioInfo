import torch
from torch.utils.data import DataLoader
from vae import VAE
from local_dataset import LocalBioData
from utils import  plot_random_samples
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print("Using:", device)

def custom_collate(batch):
    # 'batch' is a list of tensors
    # Concatenate tensors along the first dimension (dim=0)
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

# dataset
batch_size = 6 # batch 1 = onne scan file
bio_dataset = LocalBioData()
test_loader = DataLoader(dataset=bio_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
# visualize_sample(bio_dataset[214], bio_dataset.min_val, bio_dataset.max_val)

# train the VAE
model = VAE(input_dim=442, hidden_dim=221, latent_dim=110, device=device)
# load state dict
model.load_state_dict(torch.load("/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE/basic_vae/models/average_test_loss_6762696.666666667.pth", map_location=torch.device('cpu')))


# plot examples for validation
min_val = bio_dataset.min_val
max_val = bio_dataset.max_val
model.eval()
with torch.no_grad():
    for batch in test_loader:
        plot_random_samples(batch, model, device, num_samples_to_visualize=10, min_val=min_val, max_val=max_val)
        break  # We only need the first batch