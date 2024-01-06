# I want to modify the following script such that i get at the end a csv file with the following columns: normal_uncorr_train, normal_uncorr_test, normal_uncorr_val and normal_corr_train, normal_corr_test, normal_corr_val.
# I dont need these splits for the abnormal data but I want two columns for the abnormal data: abnormal_uncorr and abnormal_corr
# the content of all columns should be the full file path to the respective file
# The first step in the script should be to elicit all file in the respective categories, then we create the splits: 80% train, 10% test, 10% val 
# use pandas to create the csv file

import os
import numpy as np
import shutil

# Paths and constants
uncorr_data_path = "/prodi/hpcmem/spots_ftir_uncorr/"
corr_data_path = "/prodi/hpcmem/spots_ftir_corr/"
target_base_path = "/prodi/bioinfdata/user/bioinf3/vae_experiments/data"
sub_dirs = ["CO722"]
categories = ['normal_uncorr', 'normal_corr', 'abnormal_uncorr', 'abnormal_corr']
num_files_to_copy = 4

# Function to copy files
def copy_files(source_path, target_subfolder, label_value, num_files):
    copied_files = 0
    for sub_dir in sub_dirs:
        dir_path = os.path.join(source_path, sub_dir)
        label_path = os.path.join(dir_path, 'label.npy')
        labels = np.load(label_path)
        print

        valid_indices = np.where(labels == label_value)[0]

        # Create a list of file paths for valid data points
        for idx in valid_indices:
            if copied_files >= num_files:
                break
            file_name = f"data{idx}.npy"
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                # Define target path and copy file
                target_path = os.path.join(target_base_path, target_subfolder, file_name)
                # make directory if it does not exist
                if not os.path.exists(os.path.dirname(target_path)):
                    os.makedirs(os.path.dirname(target_path))
                print(f"Copying {file_path} to {target_path}")
                shutil.copy(file_path, target_path)
                copied_files += 1

# Main script execution
for category in categories:
    print(f"Processing category: {category}")
    if 'abnormal' in category:
        label = 1
    else:
        label = 0

    if 'uncorr' in category:
        copy_files(uncorr_data_path, category, label, num_files_to_copy)
    else:
        copy_files(corr_data_path, category, label, num_files_to_copy)

print("Script completed.")
