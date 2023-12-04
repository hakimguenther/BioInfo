import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class BioData(Dataset):
    def __init__(self):
        self.samples = []
        self.data = []

        base_dir = "/prodi/hpcmem/spots_ftir_uncorr/"
        sub_dirs = ["BC051111"]
                    # "CO722", "CO1002b", "CO1004", "CO1801a"]
        
        # Iterate through each specified subdirectory
        for sub_dir in sub_dirs:
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
                else:
                    print(f"File {file_name} not found in {dir_path}")

            for file_path in tqdm(self.samples, desc=f"Loading {sub_dir}"):
                file_data = np.load(file_path)

                try:
                    # Reshape the data such that each pixel becomes a separate sample
                    num_samples = file_data.shape[0] * file_data.shape[1]
                    file_data = file_data.reshape(num_samples, -1)

                    # Append the reshaped data to the list
                    self.data.append(file_data)
                except IndexError:
                    print(f"Skipping file {file_path} due to unexpected shape: {file_data.shape}")
                    continue  # Skip the rest of the loop and move to the next file

        # Concatenate and scale the data
        self.data = np.vstack(self.data)
        self.min_val = np.min(self.data)
        self.max_val = np.max(self.data)
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val)


    def __len__(self):
        # Return the total number of samples across all files
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sample at the specified index
        sample = self.data[idx]
        # Convert the sample to a PyTorch tensor
        return torch.tensor(sample, dtype=torch.float)
    
