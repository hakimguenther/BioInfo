import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
np.random.seed(0)
normal_rc_loss = np.random.normal(0.5, 0.1, 100)
anomalous_rc_loss = np.random.normal(0.8, 0.2, 100)
normal_kld = np.random.normal(0.3, 0.05, 100)
anomalous_kld = np.random.normal(0.6, 0.15, 100)

# Creating the plot
plt.figure(figsize=(10, 6))

# Scatter plot for normal data
plt.scatter(normal_rc_loss, normal_kld, color='blue', alpha=0.6, label='Normal Data')

# Scatter plot for anomalous data
plt.scatter(anomalous_rc_loss, anomalous_kld, color='red', alpha=0.6, label='Anomalous Data')

# Adding details to the plot
plt.title('VAE Loss Comparison: Normal vs. Anomalous Data')
plt.xlabel('Reconstruction Loss (RC Loss)')
plt.ylabel('Kullback-Leibler Divergence (KLD)')
plt.legend()

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Generating simulated data
np.random.seed(0)
normal_rc_loss = np.random.normal(0.2, 0.05, 100)
anomalous_rc_loss = np.random.normal(0.5, 0.1, 100)
normal_kld = np.random.normal(0.1, 0.02, 100)
anomalous_kld = np.random.normal(0.3, 0.06, 100)

# Calculating key statistics
normal_rc_max, normal_rc_min, normal_rc_mean = np.max(normal_rc_loss), np.min(normal_rc_loss), np.mean(normal_rc_loss)
anomalous_rc_max, anomalous_rc_min, anomalous_rc_mean = np.max(anomalous_rc_loss), np.min(anomalous_rc_loss), np.mean(anomalous_rc_loss)

normal_kld_max, normal_kld_min, normal_kld_mean = np.max(normal_kld), np.min(normal_kld), np.mean(normal_kld)
anomalous_kld_max, anomalous_kld_min, anomalous_kld_mean = np.max(anomalous_kld), np.min(anomalous_kld), np.mean(anomalous_kld)

# Preparing data for bar chart
labels = ['Max RC Loss', 'Min RC Loss', 'Mean RC Loss', 'Max KLD', 'Min KLD', 'Mean KLD']
normal_stats = [normal_rc_max, normal_rc_min, normal_rc_mean, normal_kld_max, normal_kld_min, normal_kld_mean]
anomalous_stats = [anomalous_rc_max, anomalous_rc_min, anomalous_rc_mean, anomalous_kld_max, anomalous_kld_min, anomalous_kld_mean]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

# Creating the bar chart
plt.figure(figsize=(12, 6))
rects1 = plt.bar(x - width/2, normal_stats, width, label='Normal Data')
rects2 = plt.bar(x + width/2, anomalous_stats, width, label='Anomalous Data')

# Adding details to the chart
plt.ylabel('Loss Values')
plt.title('Comparison of Key Statistics in Loss Metrics')
plt.xticks(x, labels)
plt.legend()

# Show the chart
plt.show()
