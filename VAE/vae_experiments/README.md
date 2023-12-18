## Next step:
- [x] Validate that the approach is correct
- [x] Train on server with all data for 50 epochs
- Feedback from 6.12.2023
  - track reconstruction loss & KBL seperately
    - Use KBL for training progress tracking
  - train on /prodi/hpcmem/spots_ftir_corr/ for comparison
  - look into beta-vae für gewichtung der loss terms
  - In current architecture: Add layers to the VAE
  - Normalize per pixel in get item and always take the same index for the max and min
  - Test on corrected and uncorrected data
- Todos planned 9.12:
  - [x] Find index of max and min value and hardcode them in getitem (for normalization)
  - [x] Separate the reconstruction loss and the Kulback-Leibler divergence loss into two different losses
  - [x] Use combined loss for optimization
  - [x] Implement using a validation set for validation after each epoch
  - [x] Use the KBL of the validation set for early stopping
  - [x] Run experiments on uncorrected data
  - [x] Run same experiments on corrected data
  - [ ] Compare results - how? What is a good test to determin which autoencoder is better?

## notes
- Training with fixed indices for normalization is not working:
  - I cant use cross entropy loss with fixed indices for normalization since the input and target have to be between 0 and 1
  - When I use MSE or MAE loss, the loss becomes always nan

