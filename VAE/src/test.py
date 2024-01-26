import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.loss import no_reduction_loss_function
import time
from tqdm import tqdm


####### single pixel plotting #######
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


####### multiple pixel plotting #######
def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    zs, klds, rc_losses = [], [], []
    with torch.no_grad():  # Disable gradient calculation for efficiency
        progress_bar = tqdm(loader, total=len(loader), desc=f"Evaluating model")
        for x in progress_bar:
            x = x.to(device)
            x_hat, mean, log_var, z = model(x)
            rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)
            zs.append(z.cpu().numpy())
            klds.append(kld.cpu().numpy())
            rc_losses.append(rc_loss.cpu().numpy())

    return np.concatenate(zs), np.concatenate(klds), np.concatenate(rc_losses)

def plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld, kld_scale_factor=1000):
    # Scale the KLD values for better visualization
    normal_kld_scaled = normal_kld * kld_scale_factor
    abnormal_kld_scaled = abnormal_kld * kld_scale_factor

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=300)

    # Subplot 1: Combined Data
    axes[0].scatter(normal_rc_loss, normal_kld_scaled, color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[0].scatter(abnormal_rc_loss, abnormal_kld_scaled, color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[0].set_title('Combined Data')
    axes[0].set_xlabel('Reproduction Loss (RC Loss)')
    axes[0].set_ylabel('KLD - Scaled')
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_rc_loss, normal_kld_scaled, color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[1].set_title('Normal Data Only')
    axes[1].set_xlabel('Reproduction Loss (RC Loss)')
    axes[1].set_ylabel('KLD - Scaled')
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_rc_loss, abnormal_kld_scaled, color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[2].set_title('Abnormal Data Only')
    axes[2].set_xlabel('Reproduction Loss (RC Loss)')
    axes[2].set_ylabel('KLD - Scaled')
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()
    # Capture the figure as an RGBA image and convert to a NumPy array
    fig.canvas.draw()  # Draw the canvas
    image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Capture as a string buffer and convert to NumPy array
    image_rgba = image_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to image dimensions
    
    # Convert RGBA to RGB (discard the alpha channel)
    image_rgb = image_rgba[:, :, :3]

    plt.close(fig)  # Close the figure

    return image_rgb

def plot_pca(normal_z, abnormal_z):
    # Concatenate the normal and abnormal z values and perform PCA
    all_z_values = np.vstack((normal_z, abnormal_z))  # Stack the z values vertically
    pca = PCA(n_components=2)  # Initialize PCA to reduce to 2 components
    principal_components = pca.fit_transform(all_z_values)  # Fit and transform the data

    # Split the transformed data back into normal and abnormal parts
    normal_principal_components = principal_components[:len(normal_z)]
    abnormal_principal_components = principal_components[len(normal_z):]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=300)

    # Subplot 1: Combined Data
    axes[0].scatter(normal_principal_components[:, 0], normal_principal_components[:, 1], color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[0].scatter(abnormal_principal_components[:, 0], abnormal_principal_components[:, 1], color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[0].set_title('PCA: Combined Data')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_principal_components[:, 0], normal_principal_components[:, 1], color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[1].set_title('PCA: Normal Data Only')
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_principal_components[:, 0], abnormal_principal_components[:, 1], color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[2].set_title('PCA: Abnormal Data Only')
    axes[2].set_xlabel('First Principal Component')
    axes[2].set_ylabel('Second Principal Component')
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()
    # Capture the figure as an RGBA image and convert to a NumPy array
    fig.canvas.draw()  # Draw the canvas
    image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Capture as a string buffer and convert to NumPy array
    image_rgba = image_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to image dimensions
    
    # Convert RGBA to RGB (discard the alpha channel)
    image_rgb = image_rgba[:, :, :3]

    plt.close(fig)  # Close the figure

    return image_rgb

# Main function
def test_model(model, normal_loader, abnormal_loader, docs_path, device, model_name):

    # if docs path does not exist, create it
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    print(f'{time.strftime("%Y%m%d-%H%M%S")} Evaluating {model_name}...')
    normal_z, normal_kld, normal_rc_loss = evaluate_model(model, normal_loader, device)
    abnormal_z, abnormal_kld, abnormal_rc_loss = evaluate_model(model, abnormal_loader, device)
    
    # Generate and save plots
    print(f'{time.strftime("%Y%m%d-%H%M%S")} Plotting loss scatter...')
    rc_kld_scatter = plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld)
    print(f'{time.strftime("%Y%m%d-%H%M%S")} Plotting PCA...')
    pcs_fig = plot_pca(normal_z, abnormal_z)

    # Create a 1x4 grid of subplots
    print(f'{time.strftime("%Y%m%d-%H%M%S")} Creating Combined figure...')
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), dpi=300)

    # Insert each plot into the grid
    axs[0].imshow(rc_kld_scatter)
    axs[1].imshow(pcs_fig)

    # Set titles for subplots
    axs[0].set_title('RC Loss vs KLD Scatter')
    axs[1].set_title('PCA of Z Values')

    # Remove axes for each subplot
    for ax in axs:
        ax.axis('off')

    # set overall title
    nr_normal = len(normal_z)
    nr_abnormal = len(abnormal_z)
    model_name = model_name.replace(".pth", "")
    fig.suptitle(f"Normal Samples: {nr_normal}, Abnormal Samples: {nr_abnormal}, Model: {model_name}")

    plot_path = f"{docs_path}/{model_name}_eval_plot.png"
    # if this plot already exists, change the name by adding a number
    i = 1
    while os.path.exists(plot_path):
        plot_path = f"{docs_path}/{model_name}_eval_plot_{i}.png"
        i += 1

    plt.tight_layout()
    plt.savefig(f"{docs_path}/{model_name}_eval_plot.png")  # Save the combined plot
    plt.close(fig)  # Close the figure to free up memory    

