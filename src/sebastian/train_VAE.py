import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig

from VetebraDataset import VertebraDataset


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # latent_dim * 2 because we need both mean and logvar
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Assuming input is normalized
        )

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def loss_function(recon_x, x, mean, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence between normal distribution and learned distribution
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(settings):
    # Parameters
    data_dir = settings["data_dir"]
    train_list = settings["data_set"]
    result_dir = settings["result_dir"]
    latent_dim = 20
    learning_rate = 1e-3
    batch_size = 16
    num_epochs = 50

    # Load dataset
    train_id_list_file = os.path.join(result_dir, train_list)
    train_ids = np.loadtxt(str(train_id_list_file), delimiter=",", dtype=str)
    # Initialize dataset
    dataset = VertebraDataset(
        data_dir=data_dir,
        file_list=train_ids,
        data_type='image'  # or 'mesh' or 'segmentation'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset[0].numel()  # Assuming all inputs have the same number of elements
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mean, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset)}")

    # Save the model
    model_path = os.path.join(result_dir, "vae_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-VAE')
    config = DTUConfig(args)
    if config.settings is not None:
        train_vae(config.settings)