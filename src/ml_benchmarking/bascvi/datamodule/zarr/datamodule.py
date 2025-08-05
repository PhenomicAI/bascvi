import os
import copy
import warnings
from pathlib import Path
from typing import Dict, Optional, List
import pickle
import numpy as np
import zarr
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset

# Suppress the IterableDataset warning - our implementation correctly handles worker distribution
# by properly distributing blocks among workers in _calc_start_end method
warnings.filterwarnings("ignore", message="Your `IterableDataset` has `__len__` defined")

class ZarrDataModule(pl.LightningDataModule):
    def __init__(self, data_root_dir: str = "", dataloader_args: Dict = {}, random_seed: int = 42, min_nnz: int = 300):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.random_seed = random_seed
        self.min_nnz = min_nnz
        self.backend = "zarr"  # Set backend to zarr
        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

        # Load feature presence matrix
        feature_matrix_path = os.path.join(data_root_dir, "feature_presence_matrix.npy")
        if not os.path.exists(feature_matrix_path):
            raise FileNotFoundError(f"Feature presence matrix not found at {feature_matrix_path}")
        self.feature_presence_matrix = np.load(feature_matrix_path)

        # Find all .zarr files and gene list
        block_files = sorted([str(p) for p in Path(self.data_root_dir).glob('*.zarr')])
        assert block_files, f"No .zarr files found in {self.data_root_dir}"
        z = zarr.open_group(block_files[0], mode='r')
        self.gene_list = z['var']['gene'][:].tolist()
        self.num_blocks = len(block_files)
        self.num_genes = len(self.gene_list)
        print(f'# Blocks: {self.num_blocks}\n# Genes: {self.num_genes}')

        # Load batch sizes (study/sample counts)
        with open(os.path.join(self.data_root_dir, "batch_sizes.pkl"), "rb") as f:
            max_study_idx, max_sample_idx = pickle.load(f)
        self.batch_level_sizes = [1, max_study_idx + 1, max_sample_idx + 1]

    def setup(self, stage: Optional[str] = None):
        dataset_args = dict(
            data_root_dir=self.data_root_dir,
            gene_list=self.gene_list,
            feature_presence_matrix=self.feature_presence_matrix,
            batch_level_sizes=self.batch_level_sizes,
            num_workers=self.dataloader_args.get('num_workers', 1),
            block_size=self.dataloader_args.get('batch_size', 64),
            validation_split=0.1,  # 10% for validation
            min_nnz=self.min_nnz,
        )
        if stage == "fit":
            print("Stage = Fitting")
            self.train_dataset = ZarrDataset(is_validation=False, **dataset_args)
            self.val_dataset = ZarrDataset(is_validation=True, **dataset_args)
            print(f"Train dataset length: {len(self.train_dataset)}")
            print(f"Val dataset length: {len(self.val_dataset)}")
            
        elif stage == "predict":
            print("Stage = Predicting on Zarr blocks")
            self.pred_dataset = ZarrDataset(predict_mode=True, **dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, persistent_workers=True, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, persistent_workers=True, **loader_args)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

