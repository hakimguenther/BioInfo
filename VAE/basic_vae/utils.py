import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch

def visualize_sample(tensor, min_val, max_val):
    # Ensure the tensor is on the CPU and convert it to numpy for plotting
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    # Rescale the tensor back to its original scale
    tensor = tensor * (max_val - min_val) + min_val
    data = tensor.numpy()

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(data, label='Channel Values')
    plt.xlabel('Channel')
    plt.ylabel('Value')
    plt.title('Channel Values Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()
