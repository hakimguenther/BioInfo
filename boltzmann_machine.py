import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # Initialize weights and biases
        self.weights = np.random.randn(num_visible, num_hidden) * 0.01
        self.visible_biases = np.zeros((1, num_visible))
        self.hidden_biases = np.zeros((1, num_hidden))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample(self, probs):
        return np.random.binomial(1, probs, size=probs.shape).reshape(probs.shape)
    
    def forward(self, visible_data):
        hidden_probs = self.sigmoid(visible_data @ self.weights + self.hidden_biases)
        return self.sample(hidden_probs)
    
    def backward(self, hidden_data):
        visible_probs = self.sigmoid(hidden_data @ self.weights.T + self.visible_biases)
        return self.sample(visible_probs)
    
    def contrastive_divergence(self, visible_data, learning_rate=0.1):
        visible_data = visible_data.reshape(1, -1)
        # Positive phase
        pos_hidden_probs = self.sigmoid(visible_data @ self.weights + self.hidden_biases)
        pos_hidden_activations = self.sample(pos_hidden_probs)
        
        # Negative phase
        neg_visible_probs = self.backward(pos_hidden_activations)
        neg_hidden_probs = self.sigmoid(neg_visible_probs @ self.weights + self.hidden_biases)
        
        # Weight update
        self.weights += learning_rate * (visible_data.T @ pos_hidden_probs - neg_visible_probs.T @ neg_hidden_probs)
        self.visible_biases += learning_rate * np.sum(visible_data - neg_visible_probs, axis=0)
        self.hidden_biases += learning_rate * np.sum(pos_hidden_probs - neg_hidden_probs, axis=0)
        
    def train(self, data, num_epochs=10, learning_rate=0.1):
        for epoch in range(num_epochs):
            for datum in data:
                self.contrastive_divergence(datum, learning_rate)
            print(f"Epoch {epoch + 1}/{num_epochs} completed!")

    def reconstruct(self, input_data, num_gibbs_steps=1):
        """
        Reconstructs the input data by running Gibbs sampling for a given number of steps.
        """
        reconstructed_data = input_data
        for _ in range(num_gibbs_steps):
            hidden_data = self.forward(reconstructed_data)
            reconstructed_data = self.backward(hidden_data)
        return reconstructed_data


# Diagonal pattern from top-left to bottom-right: 
# 1 0 0
# 0 1 0
# 0 0 1
pattern1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])

# Diagonal pattern from top-right to bottom-left: 
# 0 0 1
# 0 1 0
# 1 0 0
pattern2 = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0])

# Create a toy dataset repeating these patterns
dataset = [pattern1, pattern2] * 50

# Instantiate the RBM with 9 visible units (for our 3x3 images) 
# and 5 hidden units (arbitrarily chosen for this example).
rbm = RBM(num_visible=9, num_hidden=5)

# Train the RBM on the toy dataset
rbm.train(dataset, num_epochs=10)

# Create a noisy version of pattern1
# Original pattern1:
# 1 0 0
# 0 1 0
# 0 0 1
noisy_pattern = [1, 0, 0, 0, 0, 1, 0, 1, 1]  # Added some noise

# Reconstruct the noisy pattern using the trained RBM
reconstructed_data = rbm.reconstruct(noisy_pattern)

print("Noisy Input:")
print(np.array(noisy_pattern).reshape(3, 3))
print("\nReconstructed Data:")
print(np.array(reconstructed_data).reshape(3, 3))