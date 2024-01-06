import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
from tqdm import tqdm

# Utility Functions
def no_reduction_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    return reproduction_loss, KLD

def score_sample(model, sample, device):
    model.eval()  # Set the model to evaluation mode
    zs, klds, rc_losses = [], [], []
    with torch.no_grad():  # Disable gradient calculation for efficiency
        x = sample.to(device)
        x_hat, mean, log_var, z = model(x)
        rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)
        zs.append(z.cpu().numpy())
        klds.append(kld.cpu().numpy())
        rc_losses.append(rc_loss.cpu().numpy())
    return np.concatenate(zs), np.concatenate(klds), np.concatenate(rc_losses)

def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    klds, rc_losses = [], []
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for x in loader:
            x = x.to(device)
            x_hat, z, mean, log_var = model(x)
            rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)
            klds.append(kld.cpu().numpy())
            rc_losses.append(rc_loss.cpu().numpy())
    return np.concatenate(klds), np.concatenate(rc_losses)


# Plotting Functions
def plot_kld_rc_loss(normal_kld, abnormal_kld, normal_rc_loss, abnormal_rc_loss, kld_scale_factor=1000):
    # Calculate the means
    mean_normal_kld = np.mean(normal_kld) * kld_scale_factor
    mean_abnormal_kld = np.mean(abnormal_kld) * kld_scale_factor
    mean_normal_rc_loss = np.mean(normal_rc_loss)
    mean_abnormal_rc_loss = np.mean(abnormal_rc_loss)

    # Calculate the differences
    diff_kld = mean_abnormal_kld - mean_normal_kld
    diff_rc_loss = mean_abnormal_rc_loss - mean_normal_rc_loss

    # Data for plotting
    categories = ['Mean KLD Normal', 'Mean KLD Abnormal', 'Diff KLD', 'Mean RC Loss Normal', 'Mean RC Loss Abnormal', 'Diff RC Loss']
    values = [mean_normal_kld, mean_abnormal_kld, diff_kld, mean_normal_rc_loss, mean_abnormal_rc_loss, diff_rc_loss]

    # Creating the bar chart
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.bar(categories, values, color=['blue', 'red', 'green', 'blue', 'red', 'green'])
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    ax.set_xticks(range(len(categories)))  # Set fixed number of ticks
    # ax.set_xticklabels(categories, rotation=45)  # Rotate the category labels to prevent overlap
    ax.set_title('Comparison of Mean KLD and Reconstruction Loss (KLD Scaled)')
    ax.set_ylabel('Values')

    # Capture the figure as an RGBA image and convert to a NumPy array
    fig.canvas.draw()  # Draw the canvas
    image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Capture as a string buffer and convert to NumPy array
    image_rgba = image_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to image dimensions
    
    # Convert RGBA to RGB (discard the alpha channel)
    image_rgb = image_rgba[:, :, :3]

    plt.close(fig) 

    return image_rgb

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

def plot_tsne(normal_z, abnormal_z):
    # Concatenate the normal and abnormal z values and perform t-SNE
    all_z_values = np.vstack((normal_z, abnormal_z))  # Stack the z values vertically
    tsne = TSNE(n_components=2, random_state=42)  # Initialize t-SNE to reduce to 2 components
    tsne_components = tsne.fit_transform(all_z_values)  # Fit and transform the data

    # Split the transformed data back into normal and abnormal parts
    normal_tsne_components = tsne_components[:len(normal_z)]
    abnormal_tsne_components = tsne_components[len(normal_z):]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=300)

    # Subplot 1: Combined Data
    axes[0].scatter(normal_tsne_components[:, 0], normal_tsne_components[:, 1], color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[0].scatter(abnormal_tsne_components[:, 0], abnormal_tsne_components[:, 1], color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[0].set_title('t-SNE: Combined Data')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_tsne_components[:, 0], normal_tsne_components[:, 1], color='blue', alpha=0.4, s=10, label='Normal Data')
    axes[1].set_title('t-SNE: Normal Data Only')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_tsne_components[:, 0], abnormal_tsne_components[:, 1], color='red', alpha=0.4, s=10, label='Abnormal Data')
    axes[2].set_title('t-SNE: Abnormal Data Only')
    axes[2].set_xlabel('t-SNE Component 1')
    axes[2].set_ylabel('t-SNE Component 2')
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    # return the plot
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
def test_model(model, normal_loader, abnormal_loader, docs_path, device):
    # if docs path does not exist, create it
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    normal_kld, normal_rc_loss = evaluate_model(model, normal_loader, device)
    abnormal_kld, abnormal_rc_loss = evaluate_model(model, abnormal_loader, device)
    # calculate the total mean difference in kld and rc_loss
    mean_normal_kld = np.mean(normal_kld)
    mean_abnormal_kld = np.mean(abnormal_kld)
    mean_normal_rc_loss = np.mean(normal_rc_loss)
    mean_abnormal_rc_loss = np.mean(abnormal_rc_loss)
    diff_kld = mean_abnormal_kld - mean_normal_kld
    diff_rc_loss = mean_abnormal_rc_loss - mean_normal_rc_loss

    progress_bar = tqdm(enumerate(abnormal_loader), total=len(abnormal_loader), desc='Generating Plots', position=0)


    for i, abnormal_sample in progress_bar:
        # Randomly select a normal sample
        normal_sample = random.choice([x for x in normal_loader])

        # Evaluate the model on the normal and abnormal samples
        normal_z, normal_kld, normal_rc_loss = score_sample(model, normal_sample, device)
        abnormal_z, abnormal_kld, abnormal_rc_loss = score_sample(model, abnormal_sample, device)

        # Generate and save plots
        mean_fig = plot_kld_rc_loss(normal_kld, abnormal_kld, normal_rc_loss, abnormal_rc_loss)
        rc_kld_scatter = plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld)
        pcs_fig = plot_pca(normal_z, abnormal_z)
        # tsne_fig = plot_tsne(normal_z, abnormal_z)

        # Create a 1x4 grid of subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=300)

        # Insert each plot into the grid
        axs[0].imshow(mean_fig)
        axs[1].imshow(rc_kld_scatter)
        axs[2].imshow(pcs_fig)
        # axs[3].imshow(tsne_fig)

        # Set titles for subplots
        axs[0].set_title(f'Total Mean Diff KLD: {diff_kld:.6f}, Diff. RC Loss: {diff_rc_loss:.6f}')
        axs[1].set_title('RC Loss vs KLD Scatter')
        axs[2].set_title('PCA of Z Values')
        # axs[3].set_title('t-SNE of Z Values')

        # Remove axes for each subplot
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{docs_path}/combined_plots_{i}.png")  # Save the combined plot
        plt.close(fig)  # Close the figure to free up memory

        progress_bar.set_description(f"Generating Plots {i}/{len(abnormal_loader)}")
        

