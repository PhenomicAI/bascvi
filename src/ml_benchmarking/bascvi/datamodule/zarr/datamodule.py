import copy
import os
from pathlib import Path
from typing import Dict, Optional, List, Union

import pandas as pd
import pytorch_lightning as pl
import numpy as np
import anndata as ad
from torch.utils.data import DataLoader 
from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset

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
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.pretrained_batch_size = pretrained_batch_size
        self.pretrained_gene_list = pretrained_gene_list
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
            # Load from first block
            block_files = sorted([str(p) for p in Path(self.data_root_dir).glob('*.zarr')])
            adata = ad.read_zarr(block_files[0])
            self.gene_list = [str(g).lower() for g in adata.var['gene']]

        # Find all .zarr files in the train_blocks directory
        block_files = sorted([str(p) for p in Path(self.data_root_dir).glob('*.zarr')])
        assert len(block_files) > 0, f"No .zarr files found in {self.data_root_dir}"
        self.num_blocks = len(block_files)
        self.num_genes = len(self.gene_list)
        print(f'# Blocks: {self.num_blocks}')
        print(f'# Genes: {self.num_genes}')

        if stage == "fit":
            print("Stage = Fitting")
            self.train_dataset = ZarrDataset(
                data_root_dir=self.data_root_dir,
                gene_list=self.gene_list,
                num_workers=self.dataloader_args.get('num_workers', 1),
                batch_size=self.pretrained_batch_size or 64,
            )
        elif stage == "predict":
            print("Stage = Predicting on Zarr blocks")
            self.pred_dataset = ZarrDataset(
                data_root_dir=self.data_root_dir,
                gene_list=self.gene_list,
                num_workers=self.dataloader_args.get('num_workers', 1),
                batch_size=self.pretrained_batch_size or 64,
                predict_mode=True,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, **self.dataloader_args)

    def val_dataloader(self):
        # For simplicity, use a subset of blocks for validation
        val_dataset = ZarrDataset(
            data_root_dir=self.data_root_dir,
            gene_list=self.gene_list,
            num_workers=self.dataloader_args.get('num_workers', 1),
            batch_size=self.pretrained_batch_size or 64,
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

