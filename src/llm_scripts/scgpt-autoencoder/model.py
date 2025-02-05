import pytorch_lightning as pl
import torch
from torch import nn
import os
import pandas as pd
import logging

logger = logging.getLogger("pytorch_lightning")

class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),  
            nn.ReLU(True),       
            nn.Linear(256, 10), 
            nn.Sigmoid()       
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 256),  
            nn.ReLU(True),        
            nn.Linear(256, 512),  
            nn.ReLU(True)       
        )
        self.best_model_path = ""
        self.cs = nn.CosineSimilarity()

    def forward(self, x, return_embedding=False):
        encoded = self.encoder(x)
        if return_embedding:
            return encoded
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch['x']
        x_hat = self.forward(x)
        loss = self.loss(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['x']
        x_hat = self(x)
        val_loss = self.loss(x_hat, x)
        self.log('val_loss', val_loss)

    def loss(self, x, x_hat):
        return (1 - self.cs(x, x_hat)).mean(dim=-1)
    
    def predict_step(self, batch, batch_idx):
        x = batch['x']
        x_hat = self(x, return_embedding=True)
        return torch.cat((x_hat, torch.unsqueeze(batch["soma_joinid"], 1)), 1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

