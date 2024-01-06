import matplotlib.pyplot as plt
import os

def plot_losses(training_losses, validation_losses, experiment_name, docs_dir):
    epochs = range(1, len(training_losses) + 1)
    
    # Extracting loss values for plotting
    training_kdl_losses = [loss['average_training_kdl_loss'] for loss in training_losses]
    training_reproduction_losses = [loss['average_training_reproduction_loss'] for loss in training_losses]
    validation_kdl_losses = [loss['average_val_kdl_loss'] for loss in validation_losses]
    validation_reproduction_losses = [loss['average_val_reproduction_loss'] for loss in validation_losses]

    plt.figure(figsize=(12, 6))

    # Subplot for KLD loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_kdl_losses, label='Average Training KLD Loss', color='blue', marker='o')
    plt.plot(epochs, validation_kdl_losses, label='Average Validation KLD Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average KLD Loss')
    plt.title('KLD Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Subplot for MSE loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_reproduction_losses, label='Average Training MSE Loss', color='blue', marker='o')
    plt.plot(epochs, validation_reproduction_losses, label='Average Validation MSE Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.title('MSE Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Saving the plot
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    file_path = os.path.join(docs_dir, f'{experiment_name}_average_loss_plot.png')
    plt.savefig(file_path)
    plt.close()

def visualize_comparison(original_tensor, reconstructed_tensor, experiment_name, plot_dir, kdl_loss, reconstruction_loss, combined_loss):
    # Ensure both tensors are on the CPU and convert them to numpy for plotting
    original_data = original_tensor.cpu().numpy()
    reconstructed_data = reconstructed_tensor.cpu().numpy()

    # in plot_dir create a folder with the name of the experiment
    save_dir = os.path.join(plot_dir, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='red')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title(f'{experiment_name} - Combined Loss: {combined_loss:.4f}, KLD Loss: {kdl_loss:.4f}, Reconstruction Loss: {reconstruction_loss:.4f}')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(save_dir, f'{experiment_name}_comparison_{combined_loss:.4f}.png')
    plt.savefig(file_path)
    plt.close()