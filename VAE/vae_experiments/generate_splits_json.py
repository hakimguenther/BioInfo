import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Paths and constants
corr_data_path = "/prodi/hpcmem/spots_ftir_corr/"
target_base_path = "/prodi/bioinfdata/user/bioinf3/vae_experiments"
sub_dirs = ["BC051111", "CO722", "CO1002b", "CO1004", "CO1801a"]
categories = ['normal_corr', 'abnormal_corr']

# Function to list files and split data
def list_files_and_split(source_path, label_value):
    file_paths = []
    for sub_dir in sub_dirs:
        dir_path = os.path.join(source_path, sub_dir)
        label_path = os.path.join(dir_path, 'label.npy')
        labels = np.load(label_path)

        valid_indices = np.where(labels == label_value)[0]
        print("looking for label: ", label_value)
        print(labels)
        print(valid_indices)

        for idx in valid_indices:
            file_name = f"data{idx}.npy"
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                file_paths.append(file_path)

    return file_paths

# Function to create data splits
def create_splits(file_paths):
    train, test = train_test_split(file_paths, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, test, val

# Main script execution
data_dict = {column: [] for column in ['normal_corr_train', 'normal_corr_test', 'normal_corr_val',  'abnormal_corr']}

for category in categories:
    print(f"Processing category: {category}")
    if "abnormal" in category:
        label = 1
    else:
        label = 0

    file_paths = list_files_and_split(corr_data_path, label)
    
    if 'abnormal' in category:
        data_dict[category].extend(file_paths)
    else:
        train, test, val = create_splits(file_paths)
        data_dict[f'{category}_train'].extend(train)
        data_dict[f'{category}_test'].extend(test)
        data_dict[f'{category}_val'].extend(val)

# save the data_dict as a json file
with open(os.path.join(target_base_path, 'data_splits.json'), 'w') as fp:
    json.dump(data_dict, fp)