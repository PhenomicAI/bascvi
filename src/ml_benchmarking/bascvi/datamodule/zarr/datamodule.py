import copy
import os
import math
from pathlib import Path
from typing import Dict, Optional, List, Union

import pandas as pd
import pytorch_lightning as pl
import numpy as np
import zarr

from torch.utils.data import DataLoader 

from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset
from ml_benchmarking.bascvi.datamodule.zarr.utils import get_or_create_feature_presence_matrix
from ml_benchmarking.bascvi.datamodule.library_calculations import LibraryCalculator

import hashlib
import json

def load_gene_list(gene_list_path: str) -> List[str]:
    with open(gene_list_path, 'r') as f:
        genes = [line.strip().lower() for line in f if line.strip()]
    return genes


class ZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        pretrained_gene_list: Union[str, List[str]] = None,
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
        self.backend = "zarr"
        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

    def setup(self, stage: Optional[str] = None):
        # Load gene list from file if needed
        if isinstance(self.pretrained_gene_list, str):
            self.gene_list = load_gene_list(self.pretrained_gene_list)
        elif self.pretrained_gene_list is not None:
            self.gene_list = [g.lower() for g in self.pretrained_gene_list]
        else:
            z0 = zarr.open(self.file_paths[0], mode='r')
            self.gene_list = [str(g).lower() for g in z0['var']['gene'][...]]
        
        # Find zarr files
        zarr_dirs = sorted([str(p) for p in Path(self.data_root_dir).iterdir() if p.is_dir() and p.name.endswith('.zarr')])
        zarr_dirs_cache_path = os.path.join(self.root_dir, 'zarr_dirs.json')
        library_calcs_cache_path = os.path.join(self.root_dir, 'cached_calcs_and_filter', 'l_means_vars.csv')
        # Compute a hash of the zarr_dirs list for change detection
        def hash_list(lst):
            return hashlib.md5(json.dumps(lst, sort_keys=True).encode()).hexdigest()
        current_zarr_dirs_hash = hash_list(zarr_dirs)
        cached_zarr_dirs = None
        cached_zarr_dirs_hash = None
        rerun_library_calcs = False
        if os.path.exists(zarr_dirs_cache_path):
            with open(zarr_dirs_cache_path, 'r') as f:
                cached_zarr_dirs = json.load(f)
                cached_zarr_dirs_hash = hash_list(cached_zarr_dirs)
            if cached_zarr_dirs_hash != current_zarr_dirs_hash:
                rerun_library_calcs = True
        else:
            rerun_library_calcs = True
        # Save the current zarr_dirs list for future runs
        with open(zarr_dirs_cache_path, 'w') as f:
            json.dump(zarr_dirs, f)
        # Initialize library calculator for zarr data
        library_calc = LibraryCalculator(
            data_source="zarr",
            data_path=self.data_root_dir,
            root_dir=self.root_dir,
            genes_to_use=self.gene_list,
            calc_library=True,
            batch_keys={"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
        )
        def hash_path(path):
            return hashlib.md5(path.encode()).hexdigest()
        current_hashes = [hash_path(p) for p in zarr_dirs]
        if rerun_library_calcs or not os.path.exists(library_calcs_cache_path):
            library_calc.setup(zarr_dirs=zarr_dirs)
        else:
            import pandas as pd
            cached_df = pd.read_csv(library_calcs_cache_path)
            if 'zarr_path_hash' in cached_df.columns:
                cached_df = cached_df.set_index('zarr_path_hash').loc[current_hashes].reset_index()
            cached_df.to_csv(library_calcs_cache_path, index=False)
            # Assign cached_df to library_calc.library_calcs so downstream code uses the cached version
            library_calc.library_calcs = cached_df
            
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
            z = zarr.open(zarr_path, mode='r')
            self.file_paths.append(zarr_path)
            self.zarr_len_dict[zarr_path] = z['X'].attrs['shape'][0]

        if len(self.file_paths) == 0:
            raise ValueError("No .zarr files found in the provided directory.")

        self.file_paths.sort()

        if len(self.file_paths) < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = len(self.file_paths)

        self.num_genes = len(self.gene_list)

        # Set up batch information (simplified for zarr files)
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

            self.train_dataset = ZarrDataset(
                file_paths=self.file_paths,
                reference_gene_list=self.gene_list,
                zarr_len_dict=self.zarr_len_dict,
                num_batches=self.pretrained_batch_size or 64,
                num_workers=self.dataloader_args['num_workers'],
                block_size=self.block_size,
                predict_mode=False,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
            )

            val_files = self.file_paths[:max(1, len(self.file_paths) // 5)]

            self.val_dataset = ZarrDataset(
                file_paths=val_files,
                reference_gene_list=self.gene_list,
                zarr_len_dict={k: v for k, v in self.zarr_len_dict.items() if k in val_files},
                num_batches=self.pretrained_batch_size or 64,
                num_workers=self.dataloader_args['num_workers'],
                block_size=self.block_size,
                predict_mode=False,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
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
                block_size=self.block_size,
                predict_mode=True,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
            )

    def train_dataloader(self):
        """Return DataLoader for training dataset."""
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        """Return DataLoader for validation dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, persistent_workers=True, **loader_args)

    def predict_dataloader(self):
        """Return DataLoader for prediction dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, persistent_workers=True, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

