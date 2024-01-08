import numpy as np
import torch
from torch.utils.data import Dataset
import json


class BioData(Dataset):
    def __init__(self, json_path, key, trim_size=None):

        # Load the json file
        with open(json_path, 'r') as fp:
            data_dict = json.load(fp)

        self.samples = data_dict[key]
        if trim_size is not None:
            self.samples = self.samples[:trim_size]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load the file
        file_path = self.samples[idx]
        file_data = np.load(file_path)

        # Reshape the data such that each pixel becomes a separate sample
        num_samples = file_data.shape[0] * file_data.shape[1]
        file_data = file_data.reshape(num_samples, -1)

        return torch.tensor(file_data, dtype=torch.float32)
