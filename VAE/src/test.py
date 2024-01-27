import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.loss import no_reduction_loss_function
import time
from tqdm import tqdm
import io
from PIL import Image

globale_dpi = 350

####### single spectrum plotting #######
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
    best_samples = sorted_samples[:num_samples_to_visualize]
    middle_samples = sorted_samples[len(sorted_samples) // 2 - num_samples_to_visualize // 2:len(sorted_samples) // 2 + num_samples_to_visualize // 2]
    worst_samples = sorted_samples[-num_samples_to_visualize:]
    selected_samples = [best_samples, middle_samples, worst_samples]

    for samples in selected_samples:
        category = "best" if samples == best_samples else "middle" if samples == middle_samples else "worst"
        i = 0
        for comb_loss, kld_loss, rec_loss, _, x, x_hat in samples:
            i += 1
            prefix = category + "_" + str(i) + "_"
            # Flatten the tensors to 1D for visualization
            x_flat = x.view(-1)
            x_hat_flat = x_hat.view(-1)

            visualize_comparison(x_flat, x_hat_flat, experiment_name, plot_dir, kld_loss, rec_loss, comb_loss, prefix)

def visualize_comparison(original_tensor, reconstructed_tensor, experiment_name, plot_dir, kdl_loss, reconstruction_loss, combined_loss, prefix):
    # Ensure both tensors are on the CPU and convert them to numpy for plotting
    original_data = original_tensor.cpu().numpy()
    reconstructed_data = reconstructed_tensor.cpu().numpy()

    # in plot_dir create a folder with the name of the experiment
    # remove .pth from experiment_name
    experiment_name = experiment_name.replace(".pth", "")
    save_dir = os.path.join(plot_dir, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting
    plt.figure(figsize=(10, 4), dpi=globale_dpi)
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='red')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title(f'{experiment_name} - Combined Loss: {combined_loss:.6f}, KLD Loss: {kdl_loss:.6f}, Reconstruction Loss: {reconstruction_loss:.6f}')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(save_dir, f'{prefix}{experiment_name}_comparison_{combined_loss}.png')
    plt.savefig(file_path)
    plt.close()

#### evaluating model ####

def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    results = []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        progress_bar = tqdm(loader, total=len(loader), desc="Evaluating model")
        for x in progress_bar:
            x = x.to(device)
            x_hat, mean, log_var, z = model(x)
            rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)
            
            z = z.cpu().numpy()
            rc_loss = rc_loss.cpu().numpy()
            kld = kld.cpu().numpy()

            data_point = {
                "rc_loss": rc_loss,
                "kld": kld,
                "z": z
            }
            results.append(data_point)

    return results

def extract_values(data_points):
    zs, klds, rc_losses = [], [], []
    for data_point in data_points:
        zs.append(data_point['z'])
        klds.append(data_point['kld'])
        rc_losses.append(data_point['rc_loss'])

    # Stack arrays
    zs_stacked = np.vstack(zs)  # For 2D arrays with variable number of rows
    klds_stacked = np.concatenate(klds)  # For 1D arrays
    rc_losses_stacked = np.concatenate(rc_losses)  # For 1D arrays

    return zs_stacked, klds_stacked, rc_losses_stacked


#### plotting all spectra ####
def sum_batch_data(data):
    batch_aggregated = {
        'rc_loss': [],
        'kld': []
    }
    for entry in data:
        batch_aggregated['rc_loss'].append(np.sum(entry['rc_loss']))
        batch_aggregated['kld'].append(np.sum(entry['kld']))

    return batch_aggregated

def mean_batch_data(data):
    batch_aggregated = {
        'rc_loss': [],
        'kld': []
    }
    for entry in data:
        batch_aggregated['rc_loss'].append(np.mean(entry['rc_loss']))
        batch_aggregated['kld'].append(np.mean(entry['kld']))

    return batch_aggregated

def plot_summed_rc_loss_kld(normal_data, abnormal_data):
    dot_size = 30
    alpha = 1.0
    x_label = 'Summed Reconstruction Loss (MAE)'
    y_label = 'Summed KLD'
    normal_data_label = 'Normal Spectra'
    abnormal_data_label = 'Abnormal Spectra'

    # Aggregate the values by batch
    normal_aggregated = sum_batch_data(normal_data)
    abnormal_aggregated = sum_batch_data(abnormal_data)

    normal_sum_rc_loss = normal_aggregated['rc_loss']
    normal_sum_kld = normal_aggregated['kld']

    abnormal_sum_rc_loss = abnormal_aggregated['rc_loss']
    abnormal_sum_kld = abnormal_aggregated['kld']

    # Create plot
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=globale_dpi)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_sum_rc_loss, abnormal_sum_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_sum_rc_loss, normal_sum_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_sum_rc_loss, normal_sum_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[1].set_title('Normal Data Only')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_sum_rc_loss, abnormal_sum_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[2].set_title('Abnormal Data Only')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

def plot_mean_rc_loss_kld(normal_data, abnormal_data):
    dot_size = 30
    alpha = 1.0
    x_label = 'Mean Reconstruction Loss (MAE)'
    y_label = 'Mean KLD'
    normal_data_label = 'Normal Spectra'
    abnormal_data_label = 'Abnormal Spectra'

    # Aggregate the values by batch
    normal_aggregated = mean_batch_data(normal_data)
    abnormal_aggregated = mean_batch_data(abnormal_data)

    normal_mean_rc_loss = normal_aggregated['rc_loss']
    normal_mean_kld = normal_aggregated['kld']

    abnormal_mean_rc_loss = abnormal_aggregated['rc_loss']
    abnormal_mean_kld = abnormal_aggregated['kld']

    # Create plot
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=globale_dpi)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_mean_rc_loss, abnormal_mean_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_mean_rc_loss, normal_mean_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_mean_rc_loss, normal_mean_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[1].set_title('Normal Data Only')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_mean_rc_loss, abnormal_mean_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[2].set_title('Abnormal Data Only')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

#### plotting all pixels from all spectra ####

def plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld):
    dot_size = 5
    alpha = 0.5
    x_label = 'Reconstruction Loss (MAE)'
    y_label = 'KLD'
    normal_data_label = 'Normal Pixels'
    abnormal_data_label = 'Abnormal Pixels'

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=globale_dpi)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_rc_loss, abnormal_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_rc_loss, normal_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_rc_loss, normal_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[1].set_title('Normal Data Only')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_rc_loss, abnormal_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[2].set_title('Abnormal Data Only')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

def plot_pca(normal_z, abnormal_z):
    ####  Perform PCA on the normal and abnormal z values separately
    # Concatenate the normal and abnormal z values and perform PCA
    all_z_values = np.vstack((normal_z, abnormal_z))  # Stack the z values vertically
    pca = PCA(n_components=2)  # Initialize PCA to reduce to 2 components
    principal_components = pca.fit_transform(all_z_values)  # Fit and transform the data

    # Split the transformed data back into normal and abnormal parts
    normal_principal_components = principal_components[:len(normal_z)]
    abnormal_principal_components = principal_components[len(normal_z):]

    #### Create subplots
    dot_size = 5
    alpha = 0.5
    x_label = 'First Principal Component'
    y_label = 'Second Principal Component'
    normal_data_label = 'Normal Data'
    abnormal_data_label = 'Abnormal Data'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=globale_dpi)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_principal_components[:, 0], abnormal_principal_components[:, 1], color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_principal_components[:, 0], normal_principal_components[:, 1], color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[0].set_title('PCA: Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Second Principal Component')
    axes[0].legend()

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Subplot 2: Normal Data Only
    axes[1].scatter(normal_principal_components[:, 0], normal_principal_components[:, 1], color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    axes[1].set_title('PCA: Normal Data Only')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].legend()
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    # Subplot 3: Abnormal Data Only
    axes[2].scatter(abnormal_principal_components[:, 0], abnormal_principal_components[:, 1], color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[2].set_title('PCA: Abnormal Data Only')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].legend()
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)

    plt.tight_layout()

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

### Main function ####
def test_model(model, normal_loader, abnormal_loader, docs_path, device, model_name):

    # if docs path does not exist, create it
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    print(f'{time.strftime("%H:%M:%S")} Evaluating {model_name}...')
    normal_data = evaluate_model(model, normal_loader, device)
    abnormal_data = evaluate_model(model, abnormal_loader, device)

    normal_z, normal_kld, normal_rc_loss = extract_values(normal_data)
    abnormal_z, abnormal_kld, abnormal_rc_loss = extract_values(abnormal_data)

    # Generate and save plots
    print(f'{time.strftime("%H:%M:%S")} Plotting sum loss scatter spectras...')
    rc_kld_scatter_spectra_sum = plot_summed_rc_loss_kld(normal_data, abnormal_data)
    print(f'{time.strftime("%H:%M:%S")} Plotting mean loss scatter spectras...')
    rc_kld_scatter_spectra_mean = plot_mean_rc_loss_kld(normal_data, abnormal_data)
    print(f'{time.strftime("%H:%M:%S")} Plotting loss scatter pixel...')
    rc_kld_scatter_pixel = plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld)
    print(f'{time.strftime("%H:%M:%S")} Plotting PCA...')
    pcs_fig = plot_pca(normal_z, abnormal_z)

    # Create a 1x4 grid of subplots
    print(f'{time.strftime("%H:%M:%S")} Creating Combined figure...')
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), dpi=globale_dpi)

    # Insert each plot into the grid
    axs[0].imshow(rc_kld_scatter_spectra_sum)
    axs[1].imshow(rc_kld_scatter_spectra_mean)
    axs[2].imshow(rc_kld_scatter_pixel)
    axs[3].imshow(pcs_fig)

    # Set titles for subplots
    axs[0].set_title('Summed RC Loss vs Summed KLD per Spectrum')
    axs[1].set_title('Mean RC Loss vs Mean KLD per Spectrum')
    axs[2].set_title('RC Loss vs KLD per Pixel')
    axs[3].set_title('PCA of Z Values')

    # Remove axes for each subplot
    for ax in axs:
        ax.axis('off')

    # set overall title
    nr_normal = len(normal_z)
    nr_abnormal = len(abnormal_z)
    nr_normal_spec = len(normal_data)
    nr_abnormal_spec = len(abnormal_data)
    model_name = model_name.replace(".pth", "")
    fig.suptitle(f"Normal Pixels: {nr_normal}, Abnormal Pixels: {nr_abnormal}, Normal Spectra: {nr_normal_spec}, Abnormal Spectra: {nr_abnormal_spec}, Model: {model_name}")

    plot_path = f"{docs_path}/{model_name}_eval_plot.png"
    # if this plot already exists, change the name by adding a number
    i = 1
    while os.path.exists(plot_path):
        plot_path = f"{docs_path}/{model_name}_eval_plot_{i}.png"
        i += 1

    plt.tight_layout()
    plt.savefig(plot_path)  # Save the combined plot
    plt.close(fig)  # Close the figure to free up memory    

