import copy
import os
import glob
from pathlib import Path
from typing import Dict, Optional, List

import scanpy
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader 

from ml_benchmarking.bascvi.datamodule.zarr.dataset import ZarrDataset

class ZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        pretrained_gene_list: List[str] = None,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.pretrained_batch_size = pretrained_batch_size
        self.pretrained_gene_list = pretrained_gene_list

        # ensure genes are lower case
        self.pretrained_gene_list = [gene.lower() for gene in self.pretrained_gene_list]

        assert os.path.exists(data_root_dir), f"Data root directory {data_root_dir} does not exist"

    def setup(self, stage: Optional[str] = None):
        import anndata
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

        if stage == "fit":
            raise NotImplementedError("Stage = Fit not implemented for ZarrDataModule")
        elif stage == "predict":
            print("Stage = Predicting on Zarr files")
            print("# of files: ", len(self.file_paths))
            print("Pretrained batch size: ", self.pretrained_batch_size)
            self.pred_dataset = ZarrDataset(
                file_paths=self.file_paths,
                reference_gene_list=self.pretrained_gene_list,
                zarr_len_dict=self.zarr_len_dict,
                num_batches=self.pretrained_batch_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=True,
            )

    def train_dataloader(self):
        raise NotImplementedError("Training not implemented for ZarrDataModule")

    def val_dataloader(self):
        raise NotImplementedError("Training not implemented for ZarrDataModule")

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

