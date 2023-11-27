import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class BioData(Dataset):
    def __init__(self, dir_path):
        self.data = []

        print("Loading data from", dir_path)
        # Iterate through each file in the directory
        for filename in tqdm(os.listdir(dir_path)):
            if filename.endswith('.npy'):
                # Load the .npy file
                file_path = os.path.join(dir_path, filename)
                file_data = np.load(file_path)

                # Reshape the data such that each pixel becomes a separate sample
                num_samples = file_data.shape[0] * file_data.shape[1]
                file_data = file_data.reshape(num_samples, -1)

                # Append the reshaped data to the list
                self.data.append(file_data)

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