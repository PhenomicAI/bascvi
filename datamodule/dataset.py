from typing import Dict, List
import anndata
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Dataset


class ScRNASeqTorchDataset(Dataset):
    """Custom torch dataset to get data from anndata in tensor form for pytorch modules."""
       
    def __init__(
        self,
        adata: anndata.AnnData,  # scrna seq data in annData object
        batch_dict: dict,  #batch_info, null is batch_0: 1
        prod: bool = False,
        compute_lib_size: bool = False,
        predict=False,
    ):
    
        self.adata = adata
        self.batch_dict = batch_dict
        self.compute_lib_size = compute_lib_size
        self.predict = predict
        self.prod = prod
        	
    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = np.squeeze(self.adata.X[idx].toarray())
        sample = {"x": torch.from_numpy(x)}

        # Batch value for input - convert categorical to one-hot
        
        if self.predict and not self.prod:

            sample["batch_emb"] = torch.zeros(sum(self.batch_dict.values()))
        elif self.batch_dict["batch_1"] > 1:
            
            one_hots = []
            for k,v in self.batch_dict.items():
                if self.prod:
                    one_hots.append(functional.one_hot(torch.tensor(0), num_classes=v))
                else:
                    batch_ids = self.adata.obs[k].values[idx]
                    one_hots.append(functional.one_hot(torch.tensor(batch_ids), num_classes=v))
            sample["batch_emb"] = torch.cat(one_hots, 0)
            
        else:
            sample["batch_emb"] = torch.tensor([0])
            
        # Library size computation - default to smallest if two-hot used
        
        if self.compute_lib_size:
           mx = max(self.batch_dict, key=self.batch_dict.get)
           
           l_mean = torch.tensor([self.adata.obs["l_mean_"+mx].values[idx]])
           l_var = torch.tensor([self.adata.obs["l_var_"+mx].values[idx]])
           
           sample["local_l_mean"] = torch.tensor([l_mean])
           sample["local_l_var"] = torch.tensor([l_var])

        return sample
        
        
        
        
