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

from .dataset_soma import TileDBSomaTorchDataset
import tiledbsoma as soma
import tiledb

import pickle
import scanpy as sc

from tqdm import tqdm


class TileDBSomaDataModule(pl.LightningDataModule):

    def __init__(
        self,
        soma_experiment_uri,
        access_key,
        secret_key,
        rest_token,
        dataloader_args: Dict = {},
        ):
        super().__init__()

        self.ctx = soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({"rest.token": rest_token,
                                                    "vfs.s3.aws_access_key_id": access_key,
                                                    "vfs.s3.aws_secret_access_key": secret_key,
                                                    "vfs.s3.region": "us-east-2"}))

        self.soma_experiment = soma.Experiment.open(soma_experiment_uri, context=self.ctx)
        self.dataloader_args = dataloader_args




    def generate_sample_metadata(self):

        self.samples_list = sorted(self.soma_experiment.obs.read(column_names=("soma_joinid", "sample_idx",)).concat().to_pandas()["sample_idx"].unique().tolist())
        self.num_total_batches = len(self.samples_list) 
        

        if os.path.isfile("l_means_vars.csv"):
            self.library_calcs = pd.read_csv("l_means_vars.csv")
            # with open("l_means_vars.csv", "rb") as fp: 
            #     temp = pickle.load(fp)
            #     self.l_means = temp[0]
            #     self.l_vars = temp[1]
        else:
            print("generating sample metadata...")
            self.l_means = []
            self.l_vars = []
        
            for sample_idx in tqdm(self.samples_list):
                obs_table = self.soma_experiment.obs.read(
                    column_names=("soma_joinid",),
                    coords=(None, None, None, sample_idx),
                ).concat()
                row_coord = obs_table.column("soma_joinid").combine_chunks().to_numpy()

                # load new sample from TileDB
                with self.soma_experiment.axis_query(
                    measurement_name="RNA", obs_query=soma.AxisQuery(coords=(row_coord, ))
                ) as query:
                    sub_X: sc.AnnData = query.to_anndata(
                        X_name="row_norm",
                        column_names={"obs": ["soma_joinid"], "var": ["soma_joinid"]},
                    )
                    X_curr = sub_X[:, :].X
                    # calc l_mean, l_var
                    self.l_means.append(log_mean(X_curr))
                    self.l_vars.append(log_var(X_curr))

            self.library_calcs = pd.DataFrame({"sample_idx": self.samples_list, 
                                               "library_log_means": self.l_means,  
                                               "library_log_vars": self.l_vars})
            self.library_calcs.to_csv("l_means_vars.csv")



    def setup(self, stage: Optional[str] = None):

        self.generate_sample_metadata()
        
        if self.num_total_batches < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = self.num_total_batches
                    
        self.num_genes = self.soma_experiment.ms["RNA"].var.count
        self.num_cells = self.soma_experiment.obs.count
        
        print('# Samples/Batches: ', self.num_total_batches)
        print('# Genes: ', self.num_genes)
        print('# Total Cells: ', self.num_cells)  

        if stage == "fit":
            
            print("Stage = Fitting")
            
            self.val_samples = max(self.num_total_batches//10,1)
            self.train_samples = self.num_total_batches - self.val_samples
        
            print('# Samples/Batches: ', self.num_total_batches, ' # for Training: ', self.train_samples)
           
            self.train_dataset = TileDBSomaTorchDataset(self.soma_experiment,
                                                        self.samples_list[:self.train_samples],
                                                        self.num_total_batches,
                                                        self.num_genes,
                                                        self.dataloader_args['num_workers'],
                                                        self.l_means,
                                                        self.l_vars,
                                                        )
            self.val_dataset = TileDBSomaTorchDataset(self.soma_experiment,
                                                      self.samples_list[self.train_samples:],
                                                      self.num_total_batches,
                                                      self.num_genes,
                                                      self.dataloader_args['num_workers'],
                                                        self.l_means,
                                                        self.l_vars,
                                                      )
            
            
        if stage == "predict":
    
            print("Stage = Predicting")
            
            self.pred_dataset = TileDBSomaTorchDataset(self.soma_experiment, 
                                                        self.samples_list, 
                                                        self.num_total_batches,
                                                        self.num_genes,
                                                        self.dataloader_args['num_workers'],
                                                        self.l_means,
                                                        self.l_vars,
                                                        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        
        if loader_args['num_workers'] > len(self.val_dataset.samples_list):
            loader_args['num_workers'] = len(self.val_dataset.samples_list)
        
        return DataLoader(self.val_dataset, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch


def log_mean(X):
    log_counts = np.log(X.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(X):
    log_counts = np.log(X.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var

