import torch
from torch.utils.data import DataLoader
from vae import VAE_1, VAE_2
from dataset import BioData
import os
from test import test_model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

experiment_dir = "/prodi/bioinfdata/user/bioinf3/vae_experiments"
data_splits_json = os.path.join(experiment_dir, "data_splits.json")
batch_size = 4

# Test Sets
normal_dataset = BioData(data_splits_json, "normal_corr_test")
abnormal_dataset = BioData(data_splits_json, "abnormal_corr")
normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, collate_fn=custom_collate)
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, collate_fn=custom_collate)

# Paths
plot_dir = os.path.join(experiment_dir, "docs", "figures")
docs_path = os.path.join(experiment_dir, "docs", "figures", "eval_plots")


# Test VAE 2
model_name = "corr_vae_2_cont_2_best.pth"
model_path = os.path.join(experiment_dir, "models", model_name)
model = VAE_2(device=device) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# plot_good_and_bad_samples(val_loader, model, device, 5, experiment_name, plot_dir)
test_model(model, normal_loader, abnormal_loader, docs_path, device, model_name)

# Test VAE 1
model_name = "corr_vae_1_cont_2_best.pth"
model_path = os.path.join(experiment_dir, "models", model_name)
model = VAE_1(device=device) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# plot_good_and_bad_samples(val_loader, model, device, 5, experiment_name, plot_dir)
test_model(model, normal_loader, abnormal_loader, docs_path, device, model_name)
