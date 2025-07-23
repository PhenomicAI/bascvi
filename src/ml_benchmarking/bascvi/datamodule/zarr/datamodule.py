import copy
import os
import math
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import pytorch_lightning as pl
import anndata
import numpy as np

from torch.utils.data import DataLoader 

from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset
from ml_benchmarking.bascvi.datamodule.library_calculations import LibraryCalculator

class ZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        pretrained_gene_list: List[str] = None,
        root_dir: str = "",
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.pretrained_batch_size = pretrained_batch_size
        self.pretrained_gene_list = pretrained_gene_list
        self.root_dir = root_dir
        self.random_seed = random_seed

        # ensure genes are lower case
        if self.pretrained_gene_list:
            self.pretrained_gene_list = [gene.lower() for gene in self.pretrained_gene_list]

        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

    def setup(self, stage: Optional[str] = None):
        # Initialize library calculator for zarr data
        library_calc = LibraryCalculator(
            data_source="zarr",
            data_path=self.data_root_dir,
            root_dir=self.root_dir,
            genes_to_use=None,  # Will be set based on pretrained_gene_list
            calc_library=True,
            batch_keys={"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
        )
        
        # Setup library calculator
        library_calc.setup()
        
        # Get data from library calculator
        self.obs_df = library_calc.obs_df
        self.var_df = library_calc.var_df
        self.feature_presence_matrix = library_calc.feature_presence_matrix
        self.samples_list = library_calc.samples_list
        self.library_calcs = library_calc.get_library_calcs()
        
        # Set up file paths and zarr length dictionary
        self.file_paths = []
        self.zarr_len_dict = {}

        # Only find .zarr directories
        zarr_dirs = [str(p) for p in Path(self.data_root_dir).iterdir() if p.is_dir() and p.name.endswith('.zarr')]
        for zarr_path in zarr_dirs:
            ad_ = anndata.read_zarr(zarr_path)
            self.file_paths.append(zarr_path)
            self.zarr_len_dict[zarr_path] = ad_.shape[0]

        if len(self.file_paths) == 0:
            raise ValueError("No .zarr files found in the provided directory.")

        self.file_paths.sort()

        if len(self.file_paths) < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = len(self.file_paths)

        # Set up gene information
        if self.pretrained_gene_list:
            self.num_genes = len(self.pretrained_gene_list)
            self.gene_list = self.pretrained_gene_list
        else:
            # Use genes from the first zarr file
            first_adata = anndata.read_zarr(self.file_paths[0])
            self.gene_list = [gene.lower() for gene in first_adata.var_names.tolist()]
            self.num_genes = len(self.gene_list)

        # Set up batch information (simplified for zarr files)
        # For zarr files, we'll use file-based batching
        self.num_modalities = 1
        self.num_studies = 1
        self.num_samples = len(self.file_paths)
        self.batch_level_sizes = [self.num_modalities, self.num_studies, self.num_samples]
        
        # Set up soma_experiment_uri for compatibility (using data_root_dir)
        self.soma_experiment_uri = self.data_root_dir

        # Calculate total cells
        self.num_cells = sum(self.zarr_len_dict.values())
        
        # Set up block size for training
        self.block_size = min(1000, self.num_cells // 10)  # Default block size
        self.num_total_blocks = math.ceil(self.num_cells / self.block_size)

        print('# Files: ', len(self.file_paths))
        print('# Genes: ', self.num_genes)
        print('# Total Cells: ', self.num_cells)
        print('# Samples: ', self.num_samples)

        if stage == "fit":
            print("Stage = Fitting")
            # For training, we'll use all files
            self.train_dataset = ZarrDataset(
                file_paths=self.file_paths,
                reference_gene_list=self.gene_list,
                zarr_len_dict=self.zarr_len_dict,
                num_batches=self.pretrained_batch_size or 64,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=False,
            )
            # For validation, we'll use a subset of files
            val_files = self.file_paths[:max(1, len(self.file_paths) // 5)]
            self.val_dataset = ZarrDataset(
                file_paths=val_files,
                reference_gene_list=self.gene_list,
                zarr_len_dict={k: v for k, v in self.zarr_len_dict.items() if k in val_files},
                num_batches=self.pretrained_batch_size or 64,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=False,
            )
        elif stage == "predict":
            print("Stage = Predicting on Zarr files")
            print("# of files: ", len(self.file_paths))
            print("Pretrained batch size: ", self.pretrained_batch_size)
            self.pred_dataset = ZarrDataset(
                file_paths=self.file_paths,
                reference_gene_list=self.gene_list,
                zarr_len_dict=self.zarr_len_dict,
                num_batches=self.pretrained_batch_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=True,
            )

    def train_dataloader(self):
        """Return DataLoader for training dataset."""
        return DataLoader(self.train_dataset, **self.dataloader_args)

    def val_dataloader(self):
        """Return DataLoader for validation dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, **loader_args)

    def predict_dataloader(self):
        """Return DataLoader for prediction dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

