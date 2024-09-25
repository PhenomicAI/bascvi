import copy
import os
from typing import Dict, Optional, List
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
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        pretrained_gene_list: List[str] = None,
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataloader_args = dataloader_args
        self.pretrained_batch_size = pretrained_batch_size
        self.pretrained_gene_list = pretrained_gene_list

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


        if stage == "fit":
            raise NotImplementedError("Stage = Fit not implemented for AnnDataDataModule")
            
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

