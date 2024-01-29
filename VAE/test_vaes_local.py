import torch
from torch.utils.data import DataLoader
from src.vae import VAE_2_1, VAE_Enhanced
from src.dataset import BioDataScaled
import os
from src.test import test_model, plot_good_and_bad_samples
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

experiment_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE"
data_splits_json = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data/data_splits.json"


# experiment_dir = "/prodi/bioinfdata/user/bioinf3/VAE"
# data_splits_json = os.path.join(experiment_dir,"data", "data_splits.json")
batch_size = 1

# Paths
plot_dir = os.path.join(experiment_dir, "docs", "figures")
eval_plots_path = os.path.join(experiment_dir, "docs", "eval_plots")
normal_dataset = BioDataScaled(data_splits_json, "normal_corr_test")
normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, collate_fn=custom_collate)

##### Test VAE 2.1 ######
model_name = "vae_enhanced_scaled_best.pth"
output_dir = model_name.replace(".pth", "")
model_path = os.path.join(experiment_dir, "models", model_name)
model = VAE_Enhanced(device=device) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

plot_good_and_bad_samples(normal_loader, model, device, 20, output_dir +"_normal", plot_dir)

