import copy
import os
from typing import Dict, Optional
import pytorch_lightning as pl

from torch.utils.data import DataLoader 
from pathlib import Path
import scanpy
import glob
import pandas as pd

from .dataset import AnnDataDataset

class AnnDataDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str = "",
        gene_list_path: str = "",
        dataset_args: Dict = {},
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        # self.batch_keys = batch_keys
        # self.filter_genes = filter_genes
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        # self.use_l=True
        # self.batch_dict = batch_dict
        self.pretrained_batch_size = pretrained_batch_size

        with open(gene_list_path, "r") as f:
            self.reference_gene_list =  f.read().split("\n")

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


        if stage == "fit":
            raise NotImplementedError("Stage = Fit not implemented for AnnDataDataModule")
            
        elif stage == "predict":
            
            print("Stage = Predicting on AnnDatas")
            print("# of files: ", len(self.file_paths))
            print("Pretrained batch size: ", self.pretrained_batch_size)
            
            self.pred_dataset = AnnDataDataset(
                self.file_paths,
                self.reference_gene_list,
                self.adata_len_dict,
                self.pretrained_batch_size,
                self.dataloader_args['num_workers'],
                predict_mode=True,
                **self.dataset_args
            )
            

    def train_dataloader(self):
        raise NotImplementedError("Training not implemented for AnnDataDataModule")

    def val_dataloader(self):
        raise NotImplementedError("Training not implemented for AnnDataDataModule")

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

