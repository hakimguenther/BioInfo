import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BioData(Dataset):
    def __init__(self, base_dir, sub_dirs, use_labels=True):
        self.samples = []

        # Iterate through each specified subdirectory
        for sub_dir in sub_dirs:
            if use_labels:
                dir_path = os.path.join(base_dir, sub_dir)
                label_path = os.path.join(dir_path, 'label.npy')
                labels = np.load(label_path)

                # Filter out the data points with label 1
                valid_indices = np.where(labels == 0)[0]

                # Create a list of file paths for valid data points
                for idx in valid_indices:
                    file_name = f"data{idx}.npy"
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.exists(file_path):
                        self.samples.append(file_path)
                        # print(f"Added {file_path}")
                    else:
                        print(f"File {file_name} not found in {dir_path}")
            else:
                # add all npy files in the sub_dir to the samples list
                dir_path = os.path.join(base_dir, sub_dir)
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(dir_path, file_name)
                        self.samples.append(file_path)
                    else:
                        print(f"File {file_name} not found in {dir_path}")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load the file
        file_path = self.samples[idx]
        file_data = np.load(file_path)

        # Reshape the data such that each pixel becomes a separate sample
        num_samples = file_data.shape[0] * file_data.shape[1]
        file_data = file_data.reshape(num_samples, -1)

        # Scaling technique 1: min-max scaling with hard coded indices where min and max values are located
        min_values = file_data[:, 0]
        max_values = file_data[:, 356]

        # Identify rows where max is not greater than min
        valid_rows = max_values > min_values

        # if not np.all(valid_rows):
        #     print(f"Removing {np.sum(~valid_rows)} of {len(valid_rows)} invalid rows due to max <= min issue.")

        # Filter out the invalid rows
        file_data = file_data[valid_rows.flatten(), :]
        min_values = min_values[valid_rows]
        max_values = max_values[valid_rows]

        # Ensure min_values and max_values are broadcastable across all columns of file_data
        min_values = min_values[:, np.newaxis]
        max_values = max_values[:, np.newaxis]

        # Scale the data
        file_data = (file_data - min_values) / (max_values - min_values + 1e-7) # add small value to avoid division by zero

        return torch.tensor(file_data, dtype=torch.float32)

# # test the dataset locally
# base_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data"
# sub_dirs = ['normal_uncorr', 'normal_corr', 'abnormal_uncorr', 'abnormal_corr']
# for sub_dir in sub_dirs: 
#     folders = [sub_dir]
#     print(sub_dir)
#     bio_dataset = BioData(base_dir, folders, use_labels=False)
#     for i in range(len(bio_dataset)):
#         sample = bio_dataset[i]