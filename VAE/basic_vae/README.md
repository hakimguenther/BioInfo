# Variational Autoencoder
- Goal: adaptation to make the basic VAE from [this tutorial](https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f) work on our dataset


## implementation
- scaling is a problem

For your scenario, where you have a dataset of pixel values across 442 channels, you should apply normalization across the entire dataset before training. This can be done by computing the min and max values across the whole dataset and then scaling each value based on these.

Here's a general approach:

1. **Compute Min and Max Values**: Find the minimum and maximum values across your entire dataset.

2. **Scale the Data**: Apply the min-max normalization formula to scale each value to the range [0, 1]. The formula is: Scaled value = $\frac{{value - min\_value}}{{max\_value - min_value}}$

3. **Apply Scaling in the Dataset Class**: You can integrate this scaling directly into your custom dataset class.



## Research:
- (what is a VAE?)[https://jaan.io/what-is-variational-autoencoder-vae-tutorial/]
- Notes from the article:
  - 
- (explanation of VAEs)[https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73]
- Notes from the article:
  - 


### Promising links:
- https://github.com/AntixK/PyTorch-VAE