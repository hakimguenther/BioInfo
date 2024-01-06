import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Updated paths
data_path = "/Users/hannesehringfeld/SSD/Uni/Master/WS23/Bioinformatik/BioInfo/data"
categories = ['normal_uncorr', 'normal_corr', 'abnormal_uncorr', 'abnormal_corr']

# Function to list files (simplified as there are no label.npy files)
def list_files(dir_path):
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npy')]
    return file_paths

# Function to create data splits with specific requirements for normal data
def create_splits(file_paths, category):
    if 'normal' in category:
        # Ensure 2 files for train in normal categories
        train = file_paths[:2]
        test = file_paths[2:3]
        val = file_paths[3:]
    else:
        # Split as before for abnormal data
        train, test = train_test_split(file_paths, test_size=0.2, random_state=42)
        val, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, test, val

# Main script execution
data_dict = {column: [] for column in ['normal_uncorr_train', 'normal_uncorr_test', 'normal_uncorr_val', 
                                       'normal_corr_train', 'normal_corr_test', 'normal_corr_val', 
                                       'abnormal_uncorr', 'abnormal_corr']}

for category in categories:
    print(f"Processing category: {category}")
    dir_path = os.path.join(data_path, category)
    file_paths = list_files(dir_path)
    
    if 'abnormal' in category:
        data_dict[category].extend(file_paths)
    else:
        train, test, val = create_splits(file_paths, category)
        data_dict[f'{category}_train'].extend(train)
        data_dict[f'{category}_test'].extend(test)
        data_dict[f'{category}_val'].extend(val)

# Convert the dictionary to a DataFrame and save as CSV
# Find the maximum length of the lists in the dictionary
max_length = max(len(lst) for lst in data_dict.values())

# Pad shorter lists with None
for key in data_dict:
    length_difference = max_length - len(data_dict[key])
    if length_difference > 0:
        data_dict[key].extend([None] * length_difference)

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(data_dict)
csv_path = os.path.join(data_path, 'data_splits.csv')
df.to_csv(csv_path, index=False)
print(f"CSV file created at {csv_path}.")
