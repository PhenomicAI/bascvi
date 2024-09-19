import copy
import os
import time
from typing import Dict, Optional
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, ConcatDataset 
from pathlib import Path
import scanpy
import glob
import anndata
import pickle
import numpy as np
import pandas as pd

import tiledbsoma as soma
import tiledb

from dataset_emb import EmbTorchDataset

from soma_helpers import open_soma_experiment


class EmbDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        soma_experiment_uri,     
        emb_path: str,
        num_dims: int,
        cells_to_use_path: str = None,
        dataset_args: Dict = {},
        dataloader_args: Dict = {},
    ):
        super().__init__()
        

        self.soma_experiment_uri = soma_experiment_uri
        self.embeddings_df = pd.read_csv(emb_path)
        self.embeddings_df = self.embeddings_df[['soma_joinid'] + [f'embedding_{i}' for i in range(num_dims)]]
        self.num_genes = num_dims
        self.num_dims = num_dims
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.cells_to_use_path = cells_to_use_path

    def setup(self, stage: Optional[str] = None):

        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            self.obs_df = soma_experiment.obs.read(
                        column_names=("soma_joinid", "sample_idx", "dataset_idx" ,)# "nnz", )
                    ).concat().to_pandas() 

        # join the obs and embeddings
        self.obs_df = self.embeddings_df.join(self.obs_df, on="soma_joinid", how="inner", rsuffix = "_r")

        # cells to use
        if self.cells_to_use_path:
            with open(self.cells_to_use_path, "rb") as f:
                self.cells_to_use = pickle.load(f)
            print("read cell list with length ", len(self.cells_to_use))
        else:
            print("Using all cells found in obs: ", self.obs_df.shape[0])
            self.cells_to_use = self.obs_df["soma_joinid"]


        self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.cells_to_use)]


        self.samples_list = sorted(self.obs_df["sample_idx"].unique().tolist())
        self.num_samples = len(self.samples_list) 

        self.num_studies = self.obs_df.dataset_idx.max() + 1 #TODO: off by one error to fix
        self.num_batches = self.num_studies + self.num_samples

        # shuffle obs
        if stage != "predict":
            self.obs_df = self.obs_df.sample(frac=1) 
        
        print('# Batches: ', self.num_batches)
        print('# Dims: ', self.num_dims)
        print('# Cells: ', self.obs_df.shape[0])  

        if stage == "fit":
            
            print("Stage = Fitting")
            
            self.val_num_cells = max(self.obs_df.shape[0]//10, 20000)
            self.train_num_cells = self.obs_df.shape[0] - self.val_num_cells
                   
            self.train_dataset = EmbTorchDataset(
                                    self.obs_df[ : self.train_num_cells],
                                    self.num_samples,
                                    self.num_studies,
                                    self.num_genes,
                                    self.dataloader_args['num_workers'],
                                )
            self.val_dataset = EmbTorchDataset(
                                self.obs_df[self.train_num_cells : ],
                                self.num_samples,
                                self.num_studies,
                                self.num_genes,
                                self.dataloader_args['num_workers'],
                            )
            
            
        elif stage == "predict":
    
            print("Stage = Predicting")
            
            self.pred_dataset = EmbTorchDataset(
                                    self.obs_df, 
                                    self.num_samples,
                                    self.num_studies,
                                    self.num_genes,
                                    self.dataloader_args['num_workers'],
                                )
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        
        return DataLoader(self.val_dataset, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

