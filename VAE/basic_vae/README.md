# Implementation docs

- **Model:** I implemented the VAE from [this tutorial](https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f) and made it work on our dataset.
  - Review implementation in [vae.py](vae.py)
- **Data**: I used 6 scans from the dataset, loaded all into memory (one pixel, all channels). Full dataset dimension: (462316, 442) so we have 462316 examples (comming from 6 files) each have the 442 intensity values.
  - Below some examples from the training date
  - ![training_examples](basic_vae/docs/basic_training_examples_plotted.png)
  - BioInfo/tree/basic_vae/VAE/docs/basic_training_examples_plotted.png
- **Data normalization**: I normalized the data to be between 0 and 1 using the min and max values of the entire dataset. This is done in the dataset class. Review implementation in [dataset.py](dataset.py)
  - I used this formula for normalization: $\frac{{value - min\_value}}{{max\_value - min_value}}$
- **Data splits** I used a 80/20 split for training and testing datat (testing is only needed to check that the error is about the same for unseen data)
- **training** I only trained locally one epoch. See training script [train.py](train.py)
  - Below are the logs:
  - ![training_logs](docs/training_logs.png)
- **plotting examples**: I plotted a few examples from the training data and the reconstructed data. Below are the plots:
  - ![original_vs_reconstructed](basic_vae/docsoriginal_vs_reconstructed_basic.png)

## Next step:
- Validate that the approach is correct
- Train on server with all data for 50 epochs
- Read into the VAE theory in order to advance the archtecture to better capture the data


#### Interestign links for later:
- [what is a VAE?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- [explanation of VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- https://github.com/AntixK/PyTorch-VAE