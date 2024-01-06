import os
import torch
from vae import VAE_1
from torch.utils.data import DataLoader
from dataset import BioData
from test import test_model

def custom_collate(batch):
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

# Input parameters (example usage)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
experiment_name = "corr_vae_1"
experiment_dir = "/prodi/bioinfdata/user/bioinf3/vae_experiments"
# experiment_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE/vae_experiments"
csv_path = os.path.join(experiment_dir, "data_splits.csv")
# csv_path = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data/data_splits.csv"
normal_dataset = BioData(csv_path, "normal_corr_test")
abnormal_dataset = BioData(csv_path, "abnormal_corr")
docs_path = os.path.join(experiment_dir, "docs", "figures", experiment_name)
batch_size = 1

# Load the VAE
model = VAE_1(device=device).to(device)
model_dir = os.path.join(experiment_dir, "models")
model.load_state_dict(torch.load(f'{model_dir}/{experiment_name}.pth', map_location=device))

normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Call main function
test_model(model, normal_loader, abnormal_loader, docs_path, device)