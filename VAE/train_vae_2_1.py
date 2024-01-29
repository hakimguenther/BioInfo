import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from src.vae import VAE_2_1
from src.dataset import BioDataScaled
from src.loss import plot_losses, loss_function
from tqdm import tqdm
from src.earlystopper import EarlyStopper
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using:", device)

def custom_collate(batch):
    batch = [item for item in batch if item.numel() > 0]  # Filter out empty tensors
    if len(batch) == 0:
        return torch.empty(0, 442)  # Return an empty tensor with the right shape if batch is empty
    return torch.cat(batch, dim=0)

def train_epoch(model, optimizer, device, train_loader, epoch, epochs):
    model.train()
    overall_kdl_loss = 0
    overall_reproduction_loss = 0
    losses = {
        "average_training_kdl_loss": 0,
        "average_training_reproduction_loss": 0
    }
    # progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
    for x in train_loader:

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
        # progress_bar.set_description(f"Train epoch {epoch + 1}/{epochs} KDL Loss: {kld_loss.item():.4f}, Reproduction Loss: {reproduction_loss.item():.4f}")

    losses["average_training_kdl_loss"] = overall_kdl_loss / len(train_loader.dataset)
    losses["average_training_reproduction_loss"] = overall_reproduction_loss / len(train_loader.dataset)
    # print("Epoch", epoch + 1, "Average Training KDL Loss: ", losses["average_training_kdl_loss"], "Average Training Reproduction Loss: ", losses["average_training_reproduction_loss"])
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
        # print("Epoch", epoch + 1, "Average Val KDL Loss: ", losses["average_val_kdl_loss"], "Average Val Reproduction Loss: ", losses["average_val_reproduction_loss"])
    return losses

def train(model, optimizer, epochs, device, train_loader, val_loader, early_stopper, experiment_name="", docs_dir="docs", model_dir="models"):
    training_losses = []
    validation_losses = []
    progress_bar = tqdm(range(epochs), total=epochs, desc=f"Epoch 0/{epochs}")
    for epoch in progress_bar:
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
            model_dir,
            experiment_name,
            optimizer
            )
        
        # save last model
        early_stopper.save_model(model, model_dir, experiment_name + "_last", optimizer)
        
        if stop_decision:
            print("early stop")
            break

        progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")

# experiment_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE"
# data_splits_json = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data/data_splits.json"

experiment_name = "vae_2_1_scaled"
# model_name = "vae_2_1_best.pth"
experiment_dir = "/prodi/bioinfdata/user/bioinf3/VAE"
data_splits_json = os.path.join(experiment_dir, "data", "data_splits.json")
train_dataset = BioDataScaled(data_splits_json, "normal_corr_train")
val_dataset = BioDataScaled(data_splits_json, "normal_corr_val")
batch_size = 3
learning_rate = 1e-3
patience = 500
nr_epochs = 12000

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# train the VAE
# model_path = os.path.join(experiment_dir, "models", model_name)
# checkpoint = torch.load(model_path, map_location=device)

model = VAE_2_1(device=device).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
optimizer = Adam(model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
    docs_dir=os.path.join(experiment_dir, "docs", "loss_plots"),
    model_dir=os.path.join(experiment_dir, "models")
)
