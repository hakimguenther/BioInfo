import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class BioData(Dataset):
    def __init__(self, base_dir, sub_dirs, use_labels=True, search_min_max_index=False):
        self.samples = []
        self.search_min_max_index = search_min_max_index

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
                        # print(f"Added {file_path}")
                    else:
                        print(f"File {file_name} not found in {dir_path}")
        
    def find_min_max(self):
        print("Finding the indices for most frequently located min and max values...")
        list_min_indices = []
        list_max_indices = []
        min_index = 0
        max_index = 0

        # Iterate through each file in the dataset
        for file_path in tqdm(self.samples, desc="Processing Files"):

            # open the file
            file_data = np.load(file_path)
            try:
                # Reshape the data such that each pixel becomes a separate sample
                num_samples = file_data.shape[0] * file_data.shape[1]
                file_data = file_data.reshape(num_samples, -1)
                # Iterate through each row in file_data
                num_rows = file_data.shape[0]
                for i in range(num_rows):
                    row_data = file_data[i, :]
                    min_index = np.where(row_data == np.min(row_data))[0][0]
                    max_index = np.where(row_data == np.max(row_data))[0][0]
                    list_min_indices.append(min_index)
                    list_max_indices.append(max_index)

            except IndexError:
                print(f"Unexpected shape in file {file_path}: {file_data.shape}")
        # Find the most frequently located min and max values
        min_index = np.bincount(list_min_indices).argmax()
        max_index = np.bincount(list_max_indices).argmax()
        print("Most frequently located min value: ", min_index)
        print("Most frequently located max value: ", max_index)
        return min_index, max_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.search_min_max_index:
            # seach the indices for most frequently located min and max values only once
            self.min_index, self.max_index = self.find_min_max()
            self.search_min_max_index = False
        else:
            # use hard coded indices for most frequently located min and max values
            self.min_index, self.max_index = 0, 359

        # load the file
        file_path = self.samples[idx]
        file_data = np.load(file_path)

        try:
            # Reshape the data such that each pixel becomes a separate sample
            num_samples = file_data.shape[0] * file_data.shape[1]
            file_data = file_data.reshape(num_samples, -1)

            # # Scale each row by using the value of the min and max indices
            # min_values = file_data[:, self.min_index]
            # max_values = file_data[:, self.max_index]
            # min_values = min_values[:, np.newaxis]
            # max_values = max_values[:, np.newaxis]
            
            # min_values should be the minimum value per row
            min_values = np.min(file_data, axis=1)
            min_values = min_values[:, np.newaxis]
            # max_values should be the maximum value per row
            max_values = np.max(file_data, axis=1)
            max_values = max_values[:, np.newaxis]

            file_data = np.vstack(file_data)
            file_data = (file_data - min_values) / (max_values - min_values)
            # file_data = (file_data - min_values) / (max_values - min_values + 1e-10) # add small value to avoid division by zero

        except IndexError:
            print(f"Unexpected shape in file {file_path}: {file_data.shape}")
            return torch.empty(0)  # Return an empty tensor in case of shape issues
        
        # Convert the sample to a PyTorch tensor
        return torch.tensor(file_data, dtype=torch.float)

# # test the dataset locally
# base_dir = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo"
# sub_dirs = ["data"]
# bio_dataset = BioData(base_dir, sub_dirs, use_labels=False, search_min_max_index=True)
# for i in range(len(bio_dataset)):
#     sample = bio_dataset[i]

# # test the dataset on the cluster
# base_dir = "/prodi/hpcmem/spots_ftir_uncorr/"
# # sub_dirs = ["BC051111", "CO722", "CO1002b", "CO1004", "CO1801a"]
# sub_dirs = ["BC051111"]
# bio_dataset = BioData(base_dir, sub_dirs, use_labels=True, search_min_max_index=True)
# for i in range(len(bio_dataset)):
#     sample = bio_dataset[i]
