import copy
import os
import time
from typing import Dict, Optional
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader

import scanpy
import glob
import anndata
import pickle
import numpy as np

from datamodule.adata_utils import setup_anndata
from datamodule.dataset import ScRNASeqTorchDataset


class ScRNASeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",
        data_save: str = "./",
        batch_keys: list = [],
        filter_genes: set = {},
        dataset_args: Dict = {},
        dataloader_args: Dict = {},
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_save = data_save
        self.batch_keys = batch_keys
        self.filter_genes = filter_genes
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.prod=False
        self.use_l=True
    def prepare_data(self):
        # TODO: download adata files from cloud to self.data_dir
        pass

    def setup(self, stage: Optional[str] = None):
        
        ref_path = os.path.join(self.data_save, "reference_data")
            
        if not os.path.exists(ref_path):
           os.makedirs(ref_path)
        
        ## Test for existence of saved data to speed up repeats
        
        ref_adata_loc = os.path.join(ref_path, "adata.h5ad")
        ref_bdict_loc = os.path.join(ref_path, "bdict.dict")
        
        is_adata = os.path.exists(ref_adata_loc)
        is_bdict = os.path.exists(ref_bdict_loc)        
        same_data = False
        
        if is_bdict and is_adata:
            print('Saved ref data exists')
            with open(ref_bdict_loc, "rb") as fh:
                 batch_dict, batch_keys = pickle.load(fh)
                 if batch_keys == self.batch_keys:
                     self.batch_dict = batch_dict
                     same_data = True
        
        if stage is None and not same_data:
            print("Data changes or new: Loading from adata directory")
            
            # read all the .h5ad files present in data_dir into annData object
            file_paths = glob.glob(os.path.join(self.data_dir, "*.h5ad"))
            
            adatas = []
            
            if not file_paths:
                print(f"No files founf in {self.data_dir}")
            for fname in file_paths:
                print(f"Reading file {fname}")
                adatas.append(scanpy.read(fname))
            print('All files read')            
            
            full_train_adata = scanpy.concat(adatas, join="inner", index_unique=None)
            
            del adatas

            print('Writing back up')
            
            #full_train_adata.obs['age_range_years'] = full_train_adata.obs['age_range_years'].astype(str)
            
            #full_train_adata.write(ref_adata_loc)

            # filter genes
            
            #full_train_adata = scanpy.read(ref_adata_loc) 

            mask = list(map(lambda x : x[:3] not in self.filter_genes, full_train_adata.var.index.values))
            full_train_adata = full_train_adata[:,mask]

            # filter cells
            # min_genes: Minimum number of genes expressed required for a cell to pass filtering.
            
            print('Filtering for < 200 unique gene reads per cell')

            gene_counts = full_train_adata.X.getnnz(axis=1)
            mask = gene_counts > 200
            full_train_adata = full_train_adata[mask,:]

            print('Converting to Float 32')
            
            full_train_adata = full_train_adata.astype(np.float32)


            
            batch_dict = {}
            
            if self.batch_keys:
                for i,b in enumerate(self.batch_keys):
                    print('batch_id')
                    batch_id = "batch_" + str(i+1)
                    codes = full_train_adata.obs[b].astype("category").cat.codes.astype(int)
                    full_train_adata.obs[batch_id] = codes
                    batch_dict[batch_id] = codes.max() + 1
                    
            else:
                batch_dict["batch_1"] = 1
                full_train_adata.obs["batch_1"] = np.zeros(full_train_adata.obs.shape[0],)
            
            self.batch_dict = batch_dict            
            self.full_train_adata = full_train_adata
            
            self.n_genes = self.full_train_adata.shape[1]
            self.n_batches = sum(self.batch_dict.values()) 
                       
            print('Batches: ', batch_dict, ' Number of Genes: ', self.n_genes)
            
            print('Saving adata complete')
            ## save full_train_adata for faster re-run ##
            
            self.full_train_adata.write(ref_adata_loc)
            
            with open(ref_bdict_loc, "wb") as fh:
                 pickle.dump([self.batch_dict, self.batch_keys], fh)
                 
        if stage is None and same_data:
            print("No data changes: Loading from save")
            
            self.full_train_adata = scanpy.read(ref_adata_loc)
            
            self.n_genes = self.full_train_adata.shape[1]
            self.n_batches = sum(self.batch_dict.values())            
            print('Batches: ', batch_dict, ' Number of Genes: ', self.n_genes)  
            
        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit":
            print("Stage = Fitting")
            
            # batch wise library size computation
                        
            if self.dataset_args["compute_lib_size"]:
                self.full_train_adata = setup_anndata(self.full_train_adata, self.batch_keys)
            
            # split the data into train/val sets            
            idxs = range(self.full_train_adata.shape[0])
            train_idxs, val_idxs = train_test_split(idxs, test_size=0.1, random_state=42)
            
            self.train_adata = self.full_train_adata[train_idxs].copy()
            self.val_adata = self.full_train_adata[val_idxs].copy()

            self.train_dataset = ScRNASeqTorchDataset(self.train_adata, 
                                                      self.batch_dict, 
                                                      **self.dataset_args)
            self.val_dataset = ScRNASeqTorchDataset(self.val_adata, 
                                                    self.batch_dict, 
                                                    **self.dataset_args)
        if stage == "predict":

            if not self.prod:
                print("Stage = Predicting")
                self.predict_dataset = ScRNASeqTorchDataset(self.full_train_adata.copy(), 
                                                        self.batch_dict, 
                                                        **self.dataset_args)

            if self.prod: ## BaScVI only - overload datadir config option for individual fname
                print("Stage = Prod")

                adatas = []

                print(f"Reading file {self.data_dir}")
                full_train_adata = scanpy.read(self.data_dir)
            
                with open('./data/temp/reference_data/gene_ref.pkl', 'rb') as f:
                    gene_list = pickle.load(f)
                         
                mask = list(map(lambda x : x in set(gene_list), full_train_adata.var.index.values))
                full_train_adata = full_train_adata[:,mask]
                self.full_train_adata = full_train_adata[:,gene_list]
                self.batch_dict = {}

                if self.batch_keys:
                    for i,b in enumerate(self.batch_keys):
                        batch_id = "batch_" + str(i+1)
                        full_train_adata.obs[batch_id] = np.zeros(full_train_adata.shape[0],)
                        self.batch_dict[batch_id] = int(b) + 1
                        print(int(b) + 1)
                
                self.full_train_adata = setup_anndata(self.full_train_adata,self.batch_keys,prod=True)
                print(self.full_train_adata.obs.columns)
                self.predict_dataset = ScRNASeqTorchDataset(self.full_train_adata.copy(),
                                                            self.batch_dict,
                                                            prod=True,
                                                            **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        loader_args["shuffle"] = False
        return DataLoader(self.val_dataset, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        loader_args["shuffle"] = False
        return DataLoader(self.predict_dataset, **loader_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

        
