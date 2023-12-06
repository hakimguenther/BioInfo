import matplotlib.pyplot as plt
import os
import random

def visualize_sample(tensor, min_val, max_val):
    # Ensure the tensor is on the CPU and convert it to numpy for plotting
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    # Rescale the tensor back to its original scale
    tensor = tensor * (max_val - min_val) + min_val
    data = tensor.numpy()

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(data, label='Channel Values')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title('Channel Values Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_average_loss(average_losses):
    epochs = range(1, len(average_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, average_losses, label='Average Loss per Epoch', color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    save_dir = 'docs/figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'average_loss_plot.png')
    plt.savefig(file_path)
    plt.close()

def visualize_comparison(original_tensor, reconstructed_tensor, min_val, max_val, plot_id):
    # Ensure both tensors are on the CPU and convert them to numpy for plotting
    if original_tensor.device.type == 'cuda':
        original_tensor = original_tensor.cpu()
    if reconstructed_tensor.device.type == 'cuda':
        reconstructed_tensor = reconstructed_tensor.cpu()
    
    # Rescale both tensors back to their original scale
    original_tensor = original_tensor * (max_val - min_val) + min_val
    reconstructed_tensor = reconstructed_tensor * (max_val - min_val) + min_val

    original_data = original_tensor.numpy()
    reconstructed_data = reconstructed_tensor.numpy()

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='red')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title('Comparison of Original and Reconstructed Data')
    plt.legend()
    plt.grid(True)

    # Save the figure
    save_dir = 'docs/figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f'comparison_plot_{plot_id}.png')
    plt.savefig(file_path)
    plt.close()


def visualize_comparison_local(original_tensor, reconstructed_tensor, min_val, max_val, plot_id):
    # Ensure both tensors are on the CPU and convert them to numpy for plotting
    if original_tensor.device.type == 'cuda':
        original_tensor = original_tensor.cpu()
    if reconstructed_tensor.device.type == 'cuda':
        reconstructed_tensor = reconstructed_tensor.cpu()
    
    # Rescale both tensors back to their original scale
    original_tensor = original_tensor * (max_val - min_val) + min_val
    reconstructed_tensor = reconstructed_tensor * (max_val - min_val) + min_val

    original_data = original_tensor.numpy()
    reconstructed_data = reconstructed_tensor.numpy()

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='red')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title('Comparison of Original and Reconstructed Data')
    plt.legend()
    plt.grid(True)

    # Save the figure
    save_dir = '/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/VAE/basic_vae/docs/figures/local_plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f'comparison_plot_{plot_id}.png')
    plt.savefig(file_path)
    plt.close()

def plot_random_samples(batch, model, device, num_samples_to_visualize, min_val, max_val):
    # Ensure the batch is not empty
    if batch.size(0) == 0:
        print("Empty batch, no samples to display.")
        return
    
    # Select 'num_samples_to_visualize' random samples from the batch
    indices = random.sample(range(batch.size(0)), num_samples_to_visualize)

    for i in indices:
        x = batch[i].unsqueeze(0).to(device)  # Add batch dimension and move to device
        reconstructed_x, _, _ = model(x)

        # Flatten the tensors to 1D for visualization
        x_flat = x.view(-1)
        reconstructed_x_flat = reconstructed_x.view(-1)

        visualize_comparison(x_flat, reconstructed_x_flat, min_val, max_val, i)
