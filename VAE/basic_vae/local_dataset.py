import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LocalBioData(Dataset):
    def __init__(self):
        self.samples = []

        base_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo"
        sub_dirs = ["data"]
        # sub_dirs = ["BC051111"]
        
        print("Loading file paths and labels...")
        # Iterate through each specified subdirectory
        for sub_dir in sub_dirs:
            # list all npy files in the subdirectory
            file_paths = os.path.join(base_dir, sub_dir)
            file_paths = [os.path.join(file_paths, file) for file in os.listdir(file_paths) if file.endswith(".npy")]

            
            # Create a list of file paths for valid data points
            for file_path in file_paths:
                if os.path.exists(file_path):
                    self.samples.append(file_path)
                    # print(f"Added {file_path}")

        # we need to find the min and max values in the dataset to use them in the getitem function for scaling

        # Find the global min and max values
        # print("Calculating global min and max values...")
        # self.min_val = float('inf')
        # self.max_val = float('-inf')

        # for file_path in tqdm(self.samples, desc="Processing Files"):
        #     file_data = np.load(file_path)
        #     try:
        #         num_samples = file_data.shape[0] * file_data.shape[1]
        #         file_data = file_data.reshape(num_samples, -1)

        #         # Update min and max values
        #         self.min_val = min(self.min_val, np.min(file_data))
        #         self.max_val = max(self.max_val, np.max(file_data))
        #     except IndexError:
        #         print(f"Unexpected shape in file {file_path}: {file_data.shape}")
        
        # using hard coded values for min and max values (identified once for all files with label 0)
        print("Using hard coded min and max values...")
        self.min_val = -0.40619122982025146
        self.max_val = 8.0


        
        print(f"Min: {self.min_val}, Max: {self.max_val}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        file_data = np.load(file_path)
        self.data = []

        try:
            # Reshape the data such that each pixel becomes a separate sample
            num_samples = file_data.shape[0] * file_data.shape[1]
            file_data = file_data.reshape(num_samples, -1)
            self.data.append(file_data)

            # Scale the data between 0 and 1
            self.data = np.vstack(self.data)
            self.data = (self.data - self.min_val) / (self.max_val - self.min_val)
        except IndexError:
            print(f"Unexpected shape in file {file_path}: {file_data.shape}")
            return torch.empty(0)  # Return an empty tensor in case of shape issues
        

        # Convert the sample to a PyTorch tensor
        return torch.tensor(file_data, dtype=torch.float)