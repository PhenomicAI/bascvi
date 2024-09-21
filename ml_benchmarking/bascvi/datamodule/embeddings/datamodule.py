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

from .dataset import EmbTorchDataset

from ..soma.soma_helpers import open_soma_experiment

from tqdm import tqdm
from pyarrow.lib import ArrowInvalid



class EmbDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        soma_experiment_uri,       
        emb_path: str,
        num_dims: int,
        cells_to_use_path: str = None,
        barcodes_to_use_path: str = None,
        dataset_args: Dict = {},
        dataloader_args: Dict = {},
        root_dir: str = None,
        pretrained_batch_size: int = None

    ):
        super().__init__()

        self.soma_experiment_uri = soma_experiment_uri
        self.embeddings_df = pd.read_csv(emb_path)
        self.num_genes = num_dims
        self.num_dims = num_dims
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.cells_to_use_path = cells_to_use_path
        self.barcodes_to_use_path = barcodes_to_use_path
        self.pretrained_batch_size = pretrained_batch_size
        self.root_dir = root_dir
        self.gene_list = range(self.num_genes)

    def filter_and_generate_library_calcs(self):

        if os.path.isdir(os.path.join(self.root_dir, "cached_calcs_and_filter")):
            print("Loading cached metadata...")

            with open(os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_soma_ids.pkl'), 'rb') as f:
                filter_pass_soma_ids = pickle.load(f)
            print(" - loaded cached filter pass")

            self.library_calcs = pd.read_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", 'l_means_vars.csv'))
            print(" - loaded cached library calcs")

            # check if metadata calcs are done
            if max(self.library_calcs["sample_idx"].to_list()) == max(self.samples_list):
                print("   - library calcs completed!")
                self.library_calcs.set_index("sample_idx")
                self.cells_to_use = list(set(self.cells_to_use).intersection(set(filter_pass_soma_ids)))
                print(len(self.cells_to_use), " cells passed final filter.")
                return 
            else:
                print("   - resuming library calcs...")
                self.l_means = self.library_calcs["library_log_means"].to_list()
                self.l_vars = self.library_calcs["library_log_vars"].to_list()
                samples_run = self.library_calcs["sample_idx"].to_list()
        else:
            os.makedirs(os.path.join(self.root_dir, "cached_calcs_and_filter"), exist_ok=True)
            filter_pass_soma_ids = []
            self.l_means = []
            self.l_vars = []
            samples_run = []
        
            
        print("Generating sample metadata...")

        # TODO: ensure sample list is sorted

        sample_idx = self.samples_list[len(samples_run)]
        print("starting with ", sample_idx)

        for sample_idx_i in tqdm(range(len(samples_run), len(self.samples_list))):
            sample_idx = self.samples_list[sample_idx_i]
            
            # read soma_ids for this sample
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                obs_table = soma_experiment.obs.read(
                    column_names=("soma_joinid",),
                    value_filter=f"sample_idx == {sample_idx}",
                ).concat()

                row_coord = obs_table.column("soma_joinid").combine_chunks().to_numpy()
            
            # if no rows selected, return default
            if row_coord.shape[0] < 1:
                print("skipping calcs for ", sample_idx, ", not enough cells")
                self.l_means.append(0)
                self.l_vars.append(1)
                samples_run.append(sample_idx)
                self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                            "library_log_means": self.l_means,  
                                            "library_log_vars": self.l_vars})
                # save                         
                self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
                continue

            # if no counts in selected rows, return default
            try:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    X_curr = soma_experiment.ms["RNA"]["X"]["row_raw"].read((row_coord, None)).coos(shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[row_coord, :]
                    X_curr = X_curr[:, self.genes_to_use]
            except ArrowInvalid:
                print("skipping calcs for ", sample_idx, ", not enough counts")
                self.l_means.append(0)
                self.l_vars.append(1)
                samples_run.append(sample_idx)
                self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                            "library_log_means": self.l_means,  
                                            "library_log_vars": self.l_vars})
                # save                         
                self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
                continue

            # get nnz and filter
            gene_counts = X_curr.getnnz(axis=1)
            cell_mask = gene_counts > 300 
            X_curr = X_curr[cell_mask, :]
                   
            print("sample ", sample_idx, ", X shape: ", X_curr.shape)

            # calc l_mean, l_var
            self.l_means.append(log_mean(X_curr))
            self.l_vars.append(log_var(X_curr))
            samples_run.append(sample_idx)

            # apply filter mask to soma_joinids and downsample
            filter_pass_soma_ids += row_coord[cell_mask].tolist()

            # save intermediate filter pass list
            with open(os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_soma_ids.pkl'), 'wb') as f:
                pickle.dump(filter_pass_soma_ids, f)
            
            # save intermediate library calcs   
            self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                                "library_log_means": self.l_means,
                                                "library_log_vars": self.l_vars})                
            self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))

        # save                         
        self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))

        self.cells_to_use = list(set(self.cells_to_use).intersection(set(filter_pass_soma_ids)))
        print(len(self.cells_to_use), " cells passed final filter.")
        self.library_calcs.set_index("sample_idx")


    def setup(self, stage: Optional[str] = None):

        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            self.obs_df = soma_experiment.obs.read(
                        column_names=("soma_joinid", "sample_idx", "study_name", "barcode", 'dataset_idx')# "nnz", )
                    ).concat().to_pandas() 
            
            self.var_df = soma_experiment.ms["RNA"].var.read().concat().to_pandas()

        self.genes_to_use = self.var_df.index.to_list()
            
        # self.obs_df['dataset_idx'] = self.obs_df['study_name'].astype('category').cat.codes

        # join the obs and embeddings
        self.obs_df = self.embeddings_df.join(self.obs_df, on="soma_joinid", how="left", rsuffix = "_r")

        # add manual index to ensure that the order is preserved
        self.obs_df["manual_index"] = range(self.obs_df.shape[0])

        # set soma_joinid to float
        self.obs_df["soma_joinid"] = self.obs_df["soma_joinid"].astype(float)



        # cells to use
        if self.cells_to_use_path:
            with open(self.cells_to_use_path, "rb") as f:
                self.cells_to_use = pickle.load(f)
            print("read cell list with length ", len(self.cells_to_use))
        elif self.barcodes_to_use_path:
            with open(self.barcodes_to_use_path, "rb") as f:
                barcodes = pickle.load(f)
            self.cells_to_use = self.obs_df.loc[self.obs_df["barcode"].isin(barcodes)]["soma_joinid"]
            print("read cell list with length ", len(self.cells_to_use))
        else:
            print("Using all cells found in obs")
            self.cells_to_use = self.obs_df["soma_joinid"]

        self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.cells_to_use)]

        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            self.samples_list = sorted(soma_experiment.obs.read(column_names=("soma_joinid", "sample_idx",)).concat().to_pandas()["sample_idx"].unique().tolist())
            self.num_samples = len(self.samples_list) 

        # library calcs
        try:
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                self.library_calcs = soma_experiment.ms["RNA"]["sample_library_calcs"].read().concat().to_pandas()
                self.library_calcs = self.library_calcs.set_index("sample_idx")
        except: 
            self.filter_and_generate_library_calcs()
        

        self.num_studies = self.obs_df.dataset_idx.max() + 1 #TODO: off by one error to fix
        self.num_batches = self.num_studies + self.num_samples + 1 # +1 for the modality dummy

        # shuffle obs
        if stage != "predict":
            self.obs_df = self.obs_df.sample(frac=1, replace=False, random_state=42) 
            print("Shuffled obs_df")
        else:
            print("Not shuffling obs_df")

        print('# Batches: ', self.num_batches)
        print('# Dims: ', self.num_dims)
        print('# Cells: ', self.obs_df)  

        if stage == "fit":
            
            print("Stage = Fitting")
            
            self.val_num_cells = max(self.obs_df.shape[0]//10, 20000)
            self.train_num_cells = self.obs_df.shape[0] - self.val_num_cells
                   
            self.train_dataset = EmbTorchDataset(
                                    self.obs_df[ : self.train_num_cells],
                                    self.num_samples,
                                    self.num_studies,
                                    self.num_genes,
                                    self.library_calcs,
                                    self.dataloader_args['num_workers'],
                                )
            self.val_dataset = EmbTorchDataset(
                                self.obs_df[self.train_num_cells : ],
                                self.num_samples,
                                self.num_studies,
                                self.num_genes,
                                self.library_calcs,
                                self.dataloader_args['num_workers'],
                            )
            
            
        elif stage == "predict":
    
            print("Stage = Predicting")

            self.pred_dataset = EmbTorchDataset(
                                    self.obs_df, 
                                    self.num_samples,
                                    self.num_studies,
                                    self.num_genes,
                                    self.library_calcs,
                                    self.dataloader_args['num_workers'],
                                    predict_mode=True
                                )
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_args)

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

def log_mean(X):
    log_counts = np.log(X.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(X):
    log_counts = np.log(X.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var


