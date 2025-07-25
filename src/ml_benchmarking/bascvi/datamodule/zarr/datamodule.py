import copy
import os
from pathlib import Path
from typing import Dict, Optional, List, Union
import pickle
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import anndata as ad
from torch.utils.data import DataLoader 
from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset
import zarr

def load_gene_list(gene_list_path: str) -> List[str]:
    with open(gene_list_path, 'r') as f:
        genes = [line.strip().lower() for line in f if line.strip()]
    return genes

class ZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        dataloader_args: Dict = {},
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.random_seed = random_seed
        self.backend = "zarr"

        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

        # Load feature presence matrix
        feature_matrix_path = os.path.join(data_root_dir, "feature_presence_matrix.npy")
        if not os.path.exists(feature_matrix_path):
            raise FileNotFoundError(f"Feature presence matrix not found at {feature_matrix_path}")
        self.feature_presence_matrix = np.load(feature_matrix_path)

        # Find all .zarr files in the train_blocks directory
        block_files = sorted([str(p) for p in Path(self.data_root_dir).glob('*.zarr')])
        assert len(block_files) > 0, f"No .zarr files found in {self.data_root_dir}"

        z = zarr.open_group(block_files[0], mode='r')
        self.gene_list = z['var']['gene'][:].tolist()
        self.num_blocks = len(block_files)
        self.num_genes = len(self.gene_list)
        print(f'# Blocks: {self.num_blocks}')
        print(f'# Genes: {self.num_genes}')

        # Load batch sizes (study/sample counts)
        with open(os.path.join(self.data_root_dir,"batch_sizes.pkl"),"rb") as f:
            self.max_study_idx, self.max_sample_idx = pickle.load(f)
        self.num_studies = self.max_study_idx + 1
        self.num_samples = self.max_sample_idx + 1
        self.batch_level_sizes = [1, self.max_study_idx+1, self.max_sample_idx+1]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            print("Stage = Fitting")
            self.train_dataset = ZarrDataset(
                data_root_dir=self.data_root_dir,
                gene_list=self.gene_list,
                feature_presence_matrix=self.feature_presence_matrix,
                batch_level_sizes=self.batch_level_sizes,
                num_workers=self.dataloader_args.get('num_workers', 1),
                block_size=self.dataloader_args.get('batch_size', 64),
            )
        elif stage == "predict":
            print("Stage = Predicting on Zarr blocks")
            self.pred_dataset = ZarrDataset(
                data_root_dir=self.data_root_dir,
                gene_list=self.gene_list,
                feature_presence_matrix=self.feature_presence_matrix,
                batch_level_sizes=self.batch_level_sizes,
                num_workers=self.dataloader_args.get('num_workers', 1),
                predict_mode=True,
                block_size=self.dataloader_args.get('batch_size', 64),
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        # For simplicity, use a subset of blocks for validation
        val_dataset = ZarrDataset(
            data_root_dir=self.data_root_dir,
            gene_list=self.gene_list,
            feature_presence_matrix=self.feature_presence_matrix,
            batch_level_sizes=self.batch_level_sizes,
            num_workers=self.dataloader_args.get('num_workers', 1),
            block_size=self.dataloader_args.get('batch_size', 64),
        )
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(val_dataset, persistent_workers=True, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, persistent_workers=True, **loader_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

