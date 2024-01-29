import torch
from torch.utils.data import DataLoader
from src.vae import VAE_2_1
from src.dataset import BioData, BioDataScaled
import os
from VAE.src.test_without_sample_plotting import test_model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

# experiment_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE"
# data_splits_json = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data/data_splits.json"

experiment_dir = "/prodi/bioinfdata/user/bioinf3/VAE"
data_splits_json = os.path.join(experiment_dir,"data", "data_splits.json")


batch_size = 1
plot_dir = os.path.join(experiment_dir, "docs", "figures")
eval_plots_path = os.path.join(experiment_dir, "docs", "eval_plots")

##### Test VAE 2.1 unscaled ######
model_name = "vae_2_1_unscaled_final.pth"
output_dir = model_name.replace(".pth", "")
model_path = os.path.join(experiment_dir, "models", model_name)
model = VAE_2_1(device=device) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Test
abnormal_dataset = BioData(data_splits_json, "abnormal_corr")
normal_dataset = BioData(data_splits_json, "normal_corr_test")
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, collate_fn=custom_collate)
normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, collate_fn=custom_collate)
test_model(model, normal_loader, abnormal_loader, eval_plots_path, device, model_name)


##### Test VAE 2.1 unscaled ######
model_name = "vae_2_1_scaled_best.pth"
output_dir = model_name.replace(".pth", "")
model_path = os.path.join(experiment_dir, "models", model_name)
model = VAE_2_1(device=device) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Test
abnormal_dataset = BioDataScaled(data_splits_json, "abnormal_corr")
normal_dataset = BioDataScaled(data_splits_json, "normal_corr_test")
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, collate_fn=custom_collate)
normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, collate_fn=custom_collate)
test_model(model, normal_loader, abnormal_loader, eval_plots_path, device, model_name)