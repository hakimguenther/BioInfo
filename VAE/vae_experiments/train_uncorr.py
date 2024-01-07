import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from vae import VAE_1, VAE_2
from dataset import BioData
from utils import plot_losses, visualize_comparison
from tqdm import tqdm
from earlystopper import EarlyStopper
import os
from test import test_model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print("Using:", device)

def custom_collate(batch):
    # 'batch' is a list of tensors
    # Concatenate tensors along the first dimension (dim=0)
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = reproduction_loss = F.mse_loss(x_hat, x, reduction='sum')  # MSE
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss, KLD

def no_reduction_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    return reproduction_loss, KLD

def train_epoch(model, optimizer, device, train_loader, epoch, epochs):
    model.train()
    overall_kdl_loss = 0
    overall_reproduction_loss = 0
    losses = {
        "average_training_kdl_loss": 0,
        "average_training_reproduction_loss": 0
    }
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
    for x in progress_bar:

        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var, z = model(x)
        reproduction_loss, kld_loss = loss_function(x, x_hat, mean, log_var)
        
        overall_kdl_loss += kld_loss.item()
        overall_reproduction_loss += reproduction_loss.item() 

        combined_loss = reproduction_loss + kld_loss
        
        # kld_loss.backward()
        # reproduction_loss.backward()
        combined_loss.backward()
        optimizer.step()

        # Update progress bar description with the current loss
        progress_bar.set_description(f"Train epoch {epoch + 1}/{epochs} KDL Loss: {kld_loss.item():.4f}, Reproduction Loss: {reproduction_loss.item():.4f}")

    losses["average_training_kdl_loss"] = overall_kdl_loss / len(train_loader.dataset)
    losses["average_training_reproduction_loss"] = overall_reproduction_loss / len(train_loader.dataset)
    print("Epoch", epoch + 1, "Average Training KDL Loss: ", losses["average_training_kdl_loss"], "Average Training Reproduction Loss: ", losses["average_training_reproduction_loss"])
    return model, losses

def validate_checkpoint(model, device, val_loader, epoch):
    # estimate the validation loss
    model.eval()
    overall_kdl_loss = 0
    overall_reproduction_loss = 0
    losses = {
        "average_val_kdl_loss": 0,
        "average_val_reproduction_loss": 0
    }
    with torch.no_grad():
        for x in val_loader:

            x = x.to(device)

            x_hat, mean, log_var, z = model(x)
            reproduction_loss, kld_loss = loss_function(x, x_hat, mean, log_var)

            overall_kdl_loss += kld_loss.item()
            overall_reproduction_loss += reproduction_loss.item()

            # Update progress bar description with the current loss

        losses["average_val_kdl_loss"] = overall_kdl_loss / len(val_loader.dataset)
        losses["average_val_reproduction_loss"] = overall_reproduction_loss / len(val_loader.dataset)
        print("Epoch", epoch + 1, "Average Val KDL Loss: ", losses["average_val_kdl_loss"], "Average Val Reproduction Loss: ", losses["average_val_reproduction_loss"])
    return losses

def train(model, optimizer, epochs, device, train_loader, val_loader, early_stopper, experiment_name="", docs_dir="docs"):
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        # train
        model, train_losses = train_epoch(model, optimizer, device, train_loader, epoch, epochs)
        training_losses.append(train_losses)

        # validate
        val_losses = validate_checkpoint(model, device, val_loader, epoch)
        validation_losses.append(val_losses)

        # plot losses
        plot_losses(training_losses, validation_losses, experiment_name, docs_dir)

        # early stopping
        stop_decision = early_stopper.early_stop(
            val_losses["average_val_reproduction_loss"] + val_losses["average_val_kdl_loss"], 
            model,
            docs_dir,
            experiment_name
            )
        
        if stop_decision:
            print("early stop")
            break

def plot_good_and_bad_samples(val_loader, model, device, num_samples_to_visualize, experiment_name, plot_dir):
    model.eval()
    sample_losses = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device)
            x_hat, mean, log_var, z = model(x)
            reconstruction_loss, kld_loss = no_reduction_loss_function(x, x_hat, mean, log_var)
            combined_loss = reconstruction_loss + kld_loss

            # Store combined loss and corresponding indices
            for i, (comb_l, kld_l, rec_l) in enumerate(zip(combined_loss, kld_loss, reconstruction_loss)):
                sample_losses.append((comb_l.item(), kld_l.item(), rec_l.item(), i, x[i], x_hat[i]))

    # Sort by combined loss
    sorted_samples = sorted(sample_losses, key=lambda x: x[0])

    # Select top and bottom samples based on combined loss
    selected_samples = sorted_samples[:num_samples_to_visualize] + sorted_samples[-num_samples_to_visualize:]

    for comb_loss, kld_loss, rec_loss, _, x, x_hat in selected_samples:
        # Flatten the tensors to 1D for visualization
        x_flat = x.view(-1)
        x_hat_flat = x_hat.view(-1)

        visualize_comparison(x_flat, x_hat_flat, experiment_name, plot_dir, kld_loss, rec_loss, comb_loss)



############################### train vae 1 ###############################
experiment_name = "uncorr_vae_1_row_wise_scaling"
experiment_dir = "/prodi/bioinfdata/user/bioinf3/vae_experiments"
csv_path = os.path.join(experiment_dir, "data_splits.csv")
train_dataset = BioData(csv_path, "normal_uncorr_train")
val_dataset = BioData(csv_path, "normal_uncorr_val")
batch_size = 2
learning_rate = 1e-3
patience = 5
nr_epochs = 100

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# # train the VAE
model = VAE_1(device=device).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
stopper = EarlyStopper(patience=patience, min_delta=0)
train(
    model=model,
    optimizer=optimizer,
    epochs=nr_epochs,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    early_stopper=stopper,
    experiment_name=experiment_name,
    docs_dir=os.path.join(experiment_dir, "docs")
)

# load the VAE
model = VAE_1(device=device).to(device)
model_dir = os.path.join(experiment_dir, "models")
model.load_state_dict(torch.load(f'{model_dir}/{experiment_name}.pth', map_location=device))

# Evaluate the model
plot_dir = os.path.join(experiment_dir, "docs", "figures")
plot_good_and_bad_samples(val_loader, model, device, 5, experiment_name, plot_dir)

# Input parameters (example usage)
csv_path = os.path.join(experiment_dir, "data_splits.csv")
normal_dataset = BioData(csv_path, "normal_uncorr_test")
abnormal_dataset = BioData(csv_path, "abnormal_uncorr")
docs_path = os.path.join(experiment_dir, "docs", "figures", experiment_name)
batch_size = 1


normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Call main function
test_model(model, normal_loader, abnormal_loader, docs_path, device)


############################### train vae 2 ###############################
# experiment setup
experiment_name = "uncorr_vae_2_row_wise_scaling"
experiment_dir = "/prodi/bioinfdata/user/bioinf3/vae_experiments"
csv_path = os.path.join(experiment_dir, "data_splits.csv")
train_dataset = BioData(csv_path, "normal_uncorr_train")
val_dataset = BioData(csv_path, "normal_cunorr_val")
batch_size = 2
learning_rate = 1e-3
patience = 5
nr_epochs = 100

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# # train the VAE
model = VAE_2(device=device).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
stopper = EarlyStopper(patience=patience, min_delta=0)
train(
    model=model,
    optimizer=optimizer,
    epochs=nr_epochs,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    early_stopper=stopper,
    experiment_name=experiment_name,
    docs_dir=os.path.join(experiment_dir, "docs")
)

# load the VAE
model = VAE_2(device=device).to(device)
model_dir = os.path.join(experiment_dir, "models")
model.load_state_dict(torch.load(f'{model_dir}/{experiment_name}.pth', map_location=device))

# Evaluate the model
plot_dir = os.path.join(experiment_dir, "docs", "figures")
plot_good_and_bad_samples(val_loader, model, device, 5, experiment_name, plot_dir)

# Input parameters (example usage)
csv_path = os.path.join(experiment_dir, "data_splits.csv")
normal_dataset = BioData(csv_path, "normal_uncorr_test")
abnormal_dataset = BioData(csv_path, "abnormal_uncorr")
docs_path = os.path.join(experiment_dir, "docs", "figures", experiment_name)
batch_size = 1


normal_loader = DataLoader(dataset=normal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
abnormal_loader = DataLoader(dataset=abnormal_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Call main function
test_model(model, normal_loader, abnormal_loader, docs_path, device)