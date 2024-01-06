import torch
from torch import nn

class VAE_1(nn.Module):
    def __init__(self, device):
        super(VAE_1, self).__init__()

        self.device = device

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(442, 221),
            nn.LeakyReLU(0.2),
            nn.Linear(221, 110),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(110, 2)
        self.logvar_layer = nn.Linear(110, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 110),
            nn.LeakyReLU(0.2),
            nn.Linear(110, 221),
            nn.LeakyReLU(0.2),
            nn.Linear(221, 442),
            nn.Sigmoid()
            )      

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar, z
    

class VAE_2(nn.Module):
    def __init__(self, device):
        super(VAE_2, self).__init__()

        self.device = device

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(442, 221),
            nn.LeakyReLU(0.2),
            nn.Linear(221, 110),
            nn.LeakyReLU(0.2),
            nn.Linear(110, 55),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(55, 2)
        self.logvar_layer = nn.Linear(55, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 55),
            nn.LeakyReLU(0.2),
            nn.Linear(55, 110),
            nn.LeakyReLU(0.2),
            nn.Linear(110, 221),
            nn.LeakyReLU(0.2),
            nn.Linear(221, 442),
            nn.Sigmoid()
            )      

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar, z