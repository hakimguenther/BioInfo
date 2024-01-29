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
from scipy.spatial import ConvexHull
import joblib
import gc

global_dpi = 400

####### single spectrum plotting #######
def plot_good_and_bad_samples(val_loader, model, device, num_samples_to_visualize, experiment_name, plot_dir):
    print(f'{time.strftime("%H:%M:%S")} Plotting good and bad samples for {experiment_name}...')
    model.eval()
    sample_losses = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, total=len(val_loader), desc="Evaluating model")
        for batch in progress_bar:
            x = batch.to(device)
            x_hat, mean, log_var, z = model(x)
            reconstruction_loss, kld_loss = no_reduction_loss_function(x, x_hat, mean, log_var)
            combined_loss = reconstruction_loss + kld_loss

            # Move tensors to CPU to reduce GPU memory usage
            x_hat = x_hat.cpu()
            mean = mean.cpu()
            log_var = log_var.cpu()
            z = z.cpu()
            combined_loss = combined_loss.cpu()
            kld_loss = kld_loss.cpu()
            reconstruction_loss = reconstruction_loss.cpu()

            # Store combined loss and corresponding indices
            for i, (comb_l, kld_l, rec_l) in enumerate(zip(combined_loss, kld_loss, reconstruction_loss)):
                sample_losses.append((comb_l.item(), kld_l.item(), rec_l.item(), i, x[i].cpu(), x_hat[i]))

            # Clear unused GPU memory
            # torch.cuda.empty_cache()

    # Sort by combined loss
    sorted_samples = sorted(sample_losses, key=lambda x: x[0])

    # Evenly select samples from the sorted list
    step_size = max(len(sorted_samples) // num_samples_to_visualize, 1)
    selected_samples = sorted_samples[::step_size][:num_samples_to_visualize]

    i = 0
    for comb_loss, kld_loss, rec_loss, _, x, x_hat in selected_samples:
        i += 1
        prefix = str(i) + "_"
        # Flatten the tensors to 1D for visualization
        x_flat = x.view(-1)
        x_hat_flat = x_hat.view(-1)

        visualize_comparison(x_flat, x_hat_flat, experiment_name, plot_dir, kld_loss, rec_loss, comb_loss, prefix)

def visualize_comparison(original_tensor, reconstructed_tensor, experiment_name, plot_dir, kdl_loss, reconstruction_loss, combined_loss, prefix, file_name=None):
    # Ensure both tensors are on the CPU and convert them to numpy for plotting
    original_data = original_tensor
    reconstructed_data = reconstructed_tensor

    # in plot_dir create a folder with the name of the experiment
    save_dir = os.path.join(plot_dir, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if file_name is None:
        file_name = experiment_name

    # Plotting
    plt.figure(figsize=(10, 4), dpi=global_dpi)
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(reconstructed_data, label='Reconstructed Data', color='red')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title(f'{file_name} - Combined Loss: {combined_loss:.6f}, KLD Loss: {kdl_loss:.6f}, Reconstruction Loss: {reconstruction_loss:.6f}')
    plt.legend()
    plt.grid(True)

    file_path = os.path.join(save_dir, f'{prefix}{experiment_name}_comparison_{combined_loss}.png')
    plt.savefig(file_path)
    plt.close()

#### evaluating model ####

def evaluate_model_normal(model, loader, device):
    model.eval()

    zs_list, klds_list, rc_losses_list, file_paths_list = [], [], [], []

    with torch.no_grad():
        progress_bar = tqdm(loader, total=len(loader), desc="Evaluating model")
        for i, x in enumerate(progress_bar):  # Assuming each batch has data x and index idx
            x = x.to(device)
            x_hat, mean, log_var, z = model(x)
            rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)

            zs_list.append(z.cpu().numpy())
            klds_list.append(kld.cpu().numpy())
            rc_losses_list.append(rc_loss.cpu().numpy())
            file_path = loader.dataset.samples[i]
            file_paths_list.append([file_path] * x.size(0))


    # Convert lists to numpy arrays
    zs = np.vstack(zs_list)
    klds = np.concatenate(klds_list)
    rc_losses = np.concatenate(rc_losses_list)
    file_paths_list = np.concatenate(file_paths_list)

    return zs, klds, rc_losses, file_paths_list

def evaluate_model_abnormal(model, loader, device):
    model.eval()

    zs_list, klds_list, rc_losses_list, file_paths_list, x_list, x_hat_list = [], [], [], [], [], []

    with torch.no_grad():
        progress_bar = tqdm(loader, total=len(loader), desc="Evaluating model")
        for i, x in enumerate(progress_bar):  # Assuming each batch has data x and index idx
            x = x.to(device)
            x_hat, mean, log_var, z = model(x)
            rc_loss, kld = no_reduction_loss_function(x, x_hat, mean, log_var)

            zs_list.append(z.cpu().numpy())
            klds_list.append(kld.cpu().numpy())
            rc_losses_list.append(rc_loss.cpu().numpy())
            file_path = loader.dataset.samples[i]
            file_paths_list.append([file_path] * x.size(0))
            x_list.append(x.cpu().numpy())
            x_hat_list.append(x_hat.cpu().numpy())


    # Convert lists to numpy arrays
    zs = np.vstack(zs_list)
    klds = np.concatenate(klds_list)
    rc_losses = np.concatenate(rc_losses_list)
    x = np.vstack(x_list)
    x_hat = np.vstack(x_hat_list)
    file_paths_list = np.concatenate(file_paths_list)

    return zs, klds, rc_losses, file_paths_list, x, x_hat

def extract_values_normal(data_points):
    if not data_points:
        return np.array([]), np.array([]), np.array([])

    zs = data_points[0]['z']
    klds = data_points[0]['kld']
    rc_losses = data_points[0]['rc_loss']

    for data_point in data_points[1:]:
        zs = np.vstack([zs, data_point['z']])
        klds = np.concatenate([klds, data_point['kld']])
        rc_losses = np.concatenate([rc_losses, data_point['rc_loss']])

    return zs, klds, rc_losses

def extract_values_abnormal(data_points):
    if not data_points:
        return np.array([]), np.array([]), np.array([]), [], np.array([]), np.array([])

    zs = data_points[0]['z']
    klds = data_points[0]['kld']
    rc_losses = data_points[0]['rc_loss']
    x = data_points[0]['x']
    x_hat = data_points[0]['x_hat']
    file_paths = [data_points[0]['file_path']] * data_points[0]['z'].shape[0]

    for data_point in data_points[1:]:
        num_points = data_point['z'].shape[0]
        zs = np.vstack([zs, data_point['z']])
        klds = np.concatenate([klds, data_point['kld']])
        rc_losses = np.concatenate([rc_losses, data_point['rc_loss']])
        x = np.vstack([x, data_point['x']])
        x_hat = np.vstack([x_hat, data_point['x_hat']])
        file_paths.extend([data_point['file_path']] * num_points)

    return zs, klds, rc_losses, file_paths, x, x_hat

#### plotting all spectra ####
def sum_batch_data(klds, rc_losses, file_paths_list):
    # Initialize lists for start and end indices
    start_indices = [0]
    end_indices = []

    # Identify the transition points for start and end indices
    for i in range(1, len(file_paths_list)):
        if file_paths_list[i] != file_paths_list[i - 1]:
            end_indices.append(i - 1)
            start_indices.append(i)
    end_indices.append(len(file_paths_list) - 1)  # Add the end index for the last batch

    # Sum up the klds and rc_losses for each spectrum using NumPy slicing
    summed_klds = [np.sum(klds[start:end + 1]) for start, end in zip(start_indices, end_indices)]
    summed_rc_losses = [np.sum(rc_losses[start:end + 1]) for start, end in zip(start_indices, end_indices)]

    # Reduce the file_paths_list to only contain one entry per spectrum
    unique_file_paths = [file_paths_list[i] for i in start_indices]

    return summed_klds, summed_rc_losses, unique_file_paths

def mean_batch_data(klds, rc_losses, file_paths_list):
    # Initialize lists for start and end indices
    start_indices = [0]
    end_indices = []

    # Identify the transition points for start and end indices
    for i in range(1, len(file_paths_list)):
        if file_paths_list[i] != file_paths_list[i - 1]:
            end_indices.append(i - 1)
            start_indices.append(i)
    end_indices.append(len(file_paths_list) - 1)  # Add the end index for the last batch

    # Calculate mean of klds and rc_losses for each spectrum using NumPy slicing
    mean_klds = [np.mean(klds[start:end + 1]) for start, end in zip(start_indices, end_indices)]
    mean_rc_losses = [np.mean(rc_losses[start:end + 1]) for start, end in zip(start_indices, end_indices)]

    # Reduce the file_paths_list to only contain one entry per spectrum
    unique_file_paths = [file_paths_list[i] for i in start_indices]

    return mean_klds, mean_rc_losses, unique_file_paths

def plot_summed_rc_loss_kld(normal_sum_rc_loss, normal_sum_kld, abnormal_sum_rc_loss, abnormal_sum_kld, abnormal_file_paths):
    dot_size = 15
    alpha = 1.0
    x_label = 'Summed Reconstruction Loss (MAE)'
    y_label = 'Summed KLD'
    normal_data_label = 'Normal Spectra'
    abnormal_data_label = 'Abnormal Spectra'
    annotation_font_size = 15  # Set font size as small as the dots

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Calculate Convex Hull for normal data points
    normal_points = np.column_stack((normal_sum_rc_loss, normal_sum_kld))
    hull = ConvexHull(normal_points)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_sum_rc_loss, abnormal_sum_kld, color='red', alpha=0.5, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_sum_rc_loss, normal_sum_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    
    # Draw the convex hull
    for simplex in hull.simplices:
        axes[0].plot(normal_points[simplex, 0], normal_points[simplex, 1], color='green', linewidth=1)

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Label abnormal points outside the convex hull
    for i, (x, y) in enumerate(zip(abnormal_sum_rc_loss, abnormal_sum_kld)):
        if any(hull.equations[:, :2].dot(np.array([x, y])) + hull.equations[:, 2] > 0):
            # Annotate the point with its file name, directly at the point with small font size
            axes[0].annotate(abnormal_file_paths[i].split('/')[-1].replace(".npy", "").replace("data",""),
                             (x, y),
                             fontsize=annotation_font_size,  # Set the annotation font size
                             ha='center', va='center')


    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

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
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches='tight' to fit everything in
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

def plot_mean_rc_loss_kld(normal_mean_rc_loss, normal_mean_kld, abnormal_mean_rc_loss, abnormal_mean_kld, abnormal_file_paths):
    dot_size = 15
    alpha = 1.0
    x_label = 'Mean Reconstruction Loss (MAE)'
    y_label = 'Mean KLD'
    normal_data_label = 'Normal Spectra'
    abnormal_data_label = 'Abnormal Spectra'
    annotation_font_size = 15  # Set font size as small as the dots

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=global_dpi)

    # Calculate Convex Hull for normal data points
    normal_points = np.column_stack((normal_mean_rc_loss, normal_mean_kld))
    hull = ConvexHull(normal_points)


    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_mean_rc_loss, abnormal_mean_kld, color='red', alpha=0.5, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_mean_rc_loss, normal_mean_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)

    # Draw the convex hull
    for simplex in hull.simplices:
        axes[0].plot(normal_points[simplex, 0], normal_points[simplex, 1], color='green', linewidth=1)

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Label abnormal points outside the convex hull
    for i, (x, y) in enumerate(zip(abnormal_mean_rc_loss, abnormal_mean_kld)):
        if any(hull.equations[:, :2].dot(np.array([x, y])) + hull.equations[:, 2] > 0):
            # Annotate the point with its file name, directly at the point with small font size
            axes[0].annotate(abnormal_file_paths[i].split('/')[-1].replace(".npy", "").replace("data",""),
                             (x, y),
                             fontsize=annotation_font_size,  # Set the annotation font size
                             ha='center', va='center')


    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

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

def plot_rc_loss_kld_scatter(normal_rc_loss, normal_kld, abnormal_rc_loss, abnormal_kld, abnormal_file_paths, docs_path, model_name, abnormal_x, abnormal_x_hat):
    dot_size = 5
    alpha = 0.5
    annotation_font_size = 8  # Set font size as small as the dots
    x_label = 'Reconstruction Loss (MAE)'
    y_label = 'KLD'
    normal_data_label = 'Normal Pixels'
    abnormal_data_label = 'Abnormal Pixels'

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Calculate Convex Hull for normal data points
    normal_points = np.column_stack((normal_rc_loss, normal_kld))
    hull = ConvexHull(normal_points)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_rc_loss, abnormal_kld, color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_rc_loss, normal_kld, color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    
    # Draw the convex hull
    for simplex in hull.simplices:
        axes[0].plot(normal_points[simplex, 0], normal_points[simplex, 1], color='green', linewidth=1)

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()
    
    # Label abnormal points outside the convex hull
    for i, (x, y) in enumerate(zip(abnormal_rc_loss, abnormal_kld)):
        if any(hull.equations[:, :2].dot(np.array([x, y])) + hull.equations[:, 2] > 0):
            # Annotate the point with its file name, directly at the point with small font size
            file_name = abnormal_file_paths[i].split('/')[-1]
            axes[0].annotate(file_name.replace(".npy", "").replace("data",""),
                             (x, y),
                             fontsize=annotation_font_size,  # Set the annotation font size
                             ha='center', va='center')
            # call visualize_comparison for this point
            experiment_name = model_name.replace(".pth", "")
            experiment_name = experiment_name + "_outside"
            docs_dir = docs_path.replace("eval_plots", "figures")
            prefix = f"{i}_{file_name}_"

            visualize_comparison(abnormal_x[i], abnormal_x_hat[i], experiment_name, docs_dir, y, x, x+y, prefix, file_name)
    
    axes[0].set_title('Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

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
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches='tight' to fit everything in
    buf.seek(0)

    # Load the image from the buffer using PIL
    image_pil = Image.open(buf)

    # Convert the PIL Image to a NumPy array
    image_rgb = np.array(image_pil)

    plt.close(fig)  # Close the figure

    return image_rgb

def plot_pca(normal_z, abnormal_z, abnormal_file_paths):
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
    annotation_font_size = 8
    x_label = 'First Principal Component'
    y_label = 'Second Principal Component'
    normal_data_label = 'Normal Data'
    abnormal_data_label = 'Abnormal Data'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi=global_dpi)

    # Calculate Convex Hull for normal principal components
    hull = ConvexHull(normal_principal_components)

    # Subplot 1: Combined Data
    axes[0].scatter(abnormal_principal_components[:, 0], abnormal_principal_components[:, 1], color='red', alpha=alpha, s=dot_size, label=abnormal_data_label)
    axes[0].scatter(normal_principal_components[:, 0], normal_principal_components[:, 1], color='blue', alpha=alpha, s=dot_size, label=normal_data_label)
    
    # Draw the convex hull
    for simplex in hull.simplices:
        axes[0].plot(normal_principal_components[simplex, 0], normal_principal_components[simplex, 1], color='green', linewidth=1)

    # Capture the axis limits from the first plot
    xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()

    # Label abnormal points outside the convex hull
    for i, point in enumerate(abnormal_principal_components):
        if any(hull.equations[:, :2].dot(point) + hull.equations[:, 2] > 0):
            # Annotate the point with its file name
            axes[0].annotate(abnormal_file_paths[i].split('/')[-1].replace(".npy", "").replace("data", ""),
                             point,
                             fontsize=annotation_font_size,
                             ha='center', va='center')

    axes[0].set_title('PCA: Combined Data')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Second Principal Component')
    axes[0].legend()
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

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
    normal_zs, normal_klds, normal_rc_losses, normal_file_paths_list = evaluate_model_normal(model, normal_loader, device)
    abnormal_zs, abnormal_klds, abnormal_rc_losses, abnormal_file_paths_list, abnormal_x, abnormal_x_hat = evaluate_model_abnormal(model, abnormal_loader, device)

    # Extract the number of normal and abnormal pixels

    # Sum up the klds and rc_losses for each spectrum
    print(f'{time.strftime("%H:%M:%S")} Summing up loss values...')
    normal_sum_kld, normal_sum_rc_loss, normal_file_paths = sum_batch_data(normal_klds, normal_rc_losses, normal_file_paths_list)
    abnormal_sum_kld, abnormal_sum_rc_loss, abnormal_file_paths = sum_batch_data(abnormal_klds, abnormal_rc_losses, abnormal_file_paths_list)
    
    # Calculate mean of klds and rc_losses for each spectrum
    print(f'{time.strftime("%H:%M:%S")} Calculating mean values...')
    normal_mean_kld, normal_mean_rc_loss, normal_file_paths = mean_batch_data(normal_klds, normal_rc_losses, normal_file_paths_list)
    abnormal_mean_kld, abnormal_mean_rc_loss, abnormal_file_paths = mean_batch_data(abnormal_klds, abnormal_rc_losses, abnormal_file_paths_list)

    # Generating scatter plot of summed RC Loss vs KLD per spectrum
    print(f'{time.strftime("%H:%M:%S")} Plotting sum loss scatter spectras...')
    rc_kld_scatter_spectra_sum = plot_summed_rc_loss_kld(normal_sum_rc_loss, normal_sum_kld, abnormal_sum_rc_loss, abnormal_sum_kld, abnormal_file_paths)
    print(f'{time.strftime("%H:%M:%S")} Plotting mean loss scatter spectras...')
    rc_kld_scatter_spectra_mean = plot_mean_rc_loss_kld(normal_mean_rc_loss, normal_mean_kld, abnormal_mean_rc_loss, abnormal_mean_kld, abnormal_file_paths)

    # Generating scatter plot of RC Loss vs KLD per pixel
    print(f'{time.strftime("%H:%M:%S")} Plotting loss scatter pixel...')
    rc_kld_scatter_pixel = plot_rc_loss_kld_scatter(normal_rc_losses, normal_klds, abnormal_rc_losses, abnormal_klds, abnormal_file_paths_list, docs_path, model_name, abnormal_x, abnormal_x_hat)

    # Generating PCA plot
    print(f'{time.strftime("%H:%M:%S")} Plotting PCA...')
    pca_fig = plot_pca(normal_zs, abnormal_zs, abnormal_file_paths_list)

    # Extract the number of normal and abnormal spectra 
    nr_normal = len(normal_zs)
    nr_abnormal = len(abnormal_zs)
    nr_normal_spec = len(set(normal_file_paths_list))
    nr_abnormal_spec = len(set(abnormal_file_paths_list))

    # Create a 1x4 grid of subplots
    print(f'{time.strftime("%H:%M:%S")} Creating Combined figure...')
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), dpi=global_dpi)

    # Insert each plot into the grid
    axs[0].imshow(rc_kld_scatter_spectra_sum)
    axs[1].imshow(rc_kld_scatter_spectra_mean)
    axs[2].imshow(rc_kld_scatter_pixel)
    axs[3].imshow(pca_fig)

    # Set titles for subplots
    axs[0].set_title('Summed RC Loss vs Summed KLD per Spectrum')
    axs[1].set_title('Mean RC Loss vs Mean KLD per Spectrum')
    axs[2].set_title('RC Loss vs KLD per Pixel')
    axs[3].set_title('PCA of Z Values')

    # Remove axes for each subplot
    for ax in axs:
        ax.axis('off')

    # set overall title
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
