import copy
import os
import glob
from pathlib import Path
from typing import Dict, Optional, List

import scanpy
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader 

from ml_benchmarking.bascvi.datamodule.anndata.dataset import AnnDataDataset


class AnnDataDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        pretrained_gene_list: List[str] = None,
        batch_keys = {"modality": "modality", "study": "study", "sample": "sample"}
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.pretrained_batch_size = pretrained_batch_size
        self.pretrained_gene_list = pretrained_gene_list
        self.batch_keys = batch_keys

        # ensure genes are lower case
        self.pretrained_gene_list = [gene.lower() for gene in self.pretrained_gene_list]

        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

        # with open(gene_list_path, "r") as f:
        #     self.reference_gene_list =  f.read().split("\n")

    def setup(self, stage: Optional[str] = None):
            
        self.file_paths = glob.glob(os.path.join(self.data_root_dir, "*.h5ad"))

        # .h5ad
        if len(self.file_paths) > 0:
            self.adata_len_dict = {}
            for fp in self.file_paths:
                ad_ = scanpy.read(fp,backed='r')
                self.adata_len_dict[fp] = ad_.shape[0]
        
        # .mtx.gz
        else:
            print("No .h5ad files found in the provided directory, looking for *.mtx.gz* ...")
        
            fps_ = Path(self.data_root_dir).rglob('*.mtx.gz*')
            self.file_paths = [str(fp_) for fp_ in fps_]

            if len(self.file_paths) == 0:
                raise ValueError("No .h5ad or .mtx.gz files found in the provided directory.")
            else:
                self.adata_len_dict = {}
                for fp in self.file_paths:
                    ad_ = pd.read_csv(fp[:-13] + 'barcodes.tsv.gz')
                    self.adata_len_dict[fp] = ad_.shape[0]


        self.file_paths.sort()
        
        if len(self.file_paths) < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = len(self.file_paths)

        # Create batch mappings
        self.batch_mappings = {}
        self.batch_sizes = {}
        
        # Load each file to get batch information
        for fp in self.file_paths:
            adata = scanpy.read(fp)
            
            # Ensure all batch columns exist
            for key in self.batch_keys.values():
                if key not in adata.obs.columns:
                    raise ValueError(f"Column {key} not found in {fp}")
            
            # Create mappings for each batch level
            for level, col in self.batch_keys.items():
                if level not in self.batch_mappings:
                    self.batch_mappings[level] = {}
                
                # Add new categories to mapping
                for cat in adata.obs[col].unique():
                    if cat not in self.batch_mappings[level]:
                        self.batch_mappings[level][cat] = len(self.batch_mappings[level])
        
        # Set batch sizes
        for level in self.batch_keys.keys():
            self.batch_sizes[level] = len(self.batch_mappings[level])
            print(f"Number of {level}s: {self.batch_sizes[level]}")

        if stage == "fit":
            print("Stage = Fitting")
            
            # Use 20% of files for validation
            self.val_files = max(len(self.file_paths) // 5, 1)
            self.train_files = len(self.file_paths) - self.val_files
            
            print('# Files: ', len(self.file_paths), ' # for Training: ', self.train_files)
            
            self.train_dataset = AnnDataDataset(
                file_paths=self.file_paths[:self.train_files],
                reference_gene_list=self.pretrained_gene_list,
                adata_len_dict=self.adata_len_dict,
                num_batches=self.pretrained_batch_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=False,
                batch_mappings=self.batch_mappings,
                batch_sizes=self.batch_sizes,
                batch_keys=self.batch_keys
            )
            
            self.val_dataset = AnnDataDataset(
                file_paths=self.file_paths[self.train_files:],
                reference_gene_list=self.pretrained_gene_list,
                adata_len_dict=self.adata_len_dict,
                num_batches=self.pretrained_batch_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=False,
                batch_mappings=self.batch_mappings,
                batch_sizes=self.batch_sizes,
                batch_keys=self.batch_keys
            )
            
        elif stage == "predict":
            print("Stage = Predicting on AnnDatas")
            print("# of files: ", len(self.file_paths))
            print("Pretrained batch size: ", self.pretrained_batch_size)
            
            self.pred_dataset = AnnDataDataset(
                file_paths=self.file_paths,
                reference_gene_list=self.pretrained_gene_list,
                adata_len_dict=self.adata_len_dict,
                num_batches=self.pretrained_batch_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=True,
                batch_mappings=self.batch_mappings,
                batch_sizes=self.batch_sizes,
                batch_keys=self.batch_keys
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, persistent_workers=True, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

