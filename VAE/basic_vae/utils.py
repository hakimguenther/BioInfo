import matplotlib.pyplot as plt
import os

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

    print(f"Saved plot to {file_path}")
