import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import lightning as L

class CNN(L.LightningModule):
    """
    A sample 2D CNN model.
    """
    
    def __init__(self, lr=0.001, classification=False):
        super().__init__()
        
        self.lr = lr 
        
        # Model architecture
        self.net = []
        self.net.append(nn.Conv2d(2, 3, 3))
        self.net.append(models.resnet50(weights=models.ResNet50_Weights.DEFAULT))
        self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)

        self.linear = nn.Linear(1000, 1)

    def features(self, X):

        return self.net(X)
        
    def forward(self, X):
        
        features = self.net(X)
        prediction = self.linear(features)

        return prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        # Split batch
        X, y = batch
        
        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log(f"train/loss", loss)

        accuracy = torch.sum((F.sigmoid(y_hat) > 0.5) == (F.sigmoid(y) > 0.5)) / X.shape[0]
        self.log(f"train/accuracy", accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        # Split batch
        X, y = batch
        
        # Forward pass
        y_hat = self(X)
        
        # Compute loss
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log(f"val/loss", loss)
        
        accuracy = torch.sum((F.sigmoid(y_hat) > 0.5) == (F.sigmoid(y) > 0.5)) / X.shape[0]
        self.log(f"val/accuracy", accuracy)
