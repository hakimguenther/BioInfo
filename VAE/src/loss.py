import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

## loss functions
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.l1_loss(x_hat, x, reduction='none').mean(dim=1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    reproduction_loss = reproduction_loss.mean()
    KLD = KLD.mean()
    return reproduction_loss, KLD

def no_reduction_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.l1_loss(x_hat, x, reduction='none').sum(dim=1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    return reproduction_loss, KLD

def plot_losses(training_losses, validation_losses, experiment_name, docs_dir):
    epochs = range(1, len(training_losses) + 1)
    
    # Extracting loss values for plotting
    training_kdl_losses = [loss['average_training_kdl_loss'] for loss in training_losses]
    training_reproduction_losses = [loss['average_training_reproduction_loss'] for loss in training_losses]
    validation_kdl_losses = [loss['average_val_kdl_loss'] for loss in validation_losses]
    validation_reproduction_losses = [loss['average_val_reproduction_loss'] for loss in validation_losses]

    # lowest val losses
    val_losses = [validation_kdl_losses[i] + validation_reproduction_losses[i] for i in range(len(validation_kdl_losses))]
    epoch_with_lowest_val_loss = val_losses.index(min(val_losses)) + 1
    lowest_val_kdl_loss = validation_kdl_losses[epoch_with_lowest_val_loss - 1]
    lowest_val_reproduction_loss = validation_reproduction_losses[epoch_with_lowest_val_loss - 1]

    plt.figure(figsize=(15, 6))

    # Subplot for KLD loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_kdl_losses, label='Average Training KLD Loss', color='blue', marker='o')
    plt.plot(epochs, validation_kdl_losses, label='Average Validation KLD Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average KLD Loss')
    plt.title('KLD Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.text(epoch_with_lowest_val_loss, lowest_val_kdl_loss, f'{lowest_val_kdl_loss:.6f}')

    # Subplot for MAE loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_reproduction_losses, label='Average Training MAE Loss', color='blue', marker='o')
    plt.plot(epochs, validation_reproduction_losses, label='Average Validation MAE Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average MAE Loss (Log Scale)')
    plt.title('MAE Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.text(epoch_with_lowest_val_loss, lowest_val_reproduction_loss, f'{lowest_val_reproduction_loss:.6f}')

    # set overall title
    plt.suptitle(f'{experiment_name} - Best val KLD Loss: {lowest_val_kdl_loss:.6f}, Best val MAE (RC) Loss: {lowest_val_reproduction_loss:.6f}, Combined Loss: {lowest_val_kdl_loss + lowest_val_reproduction_loss:.6f} at epoch {epoch_with_lowest_val_loss}')

    # Saving the plot
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    file_path = os.path.join(docs_dir, f'{experiment_name}_average_loss_plot.png')
    plt.savefig(file_path)
    plt.close()

