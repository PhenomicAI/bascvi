from typing import Dict, List
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Dataset
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix
import tiledbsoma as soma
import scanpy as sc
import math



class EmbTorchDataset(Dataset):
    """Custom torch dataset to get data from tiledbsoma in tensor form for pytorch modules."""
       
    def __init__(
        self,
        obs_df,
        num_samples,
        num_studies,
        num_dims,
        library_calcs,
        num_workers,
        pretrained_batch_size = None,
        predict_mode = False,
    ):     

        self.obs_df = obs_df

        self.num_samples = num_samples
        self.num_studies = num_studies
        self.num_batches = num_samples + num_studies

        self.predict_mode = predict_mode

        if pretrained_batch_size is not None:
            self.num_batches = pretrained_batch_size
            self.predict_mode = True

        self.num_dims = num_dims

        self.cell_counter = 0

        self.library_calcs = library_calcs

        self.num_workers = num_workers

        self._len = self.obs_df.shape[0]

        print("len dataset:", self._len)

    def __len__(self):
        return self._len

    
    def __getitem__(self, idx):

        # get row from obs_df
        row = self.obs_df.iloc[idx]

        # get soma joinid
        soma_joinid = row["soma_joinid"]

        # get manual index
        manual_index = row["manual_index"]

        # get x
        try:
            x = row[["embedding_" + str(dim) for dim in range(self.num_dims)]].values
        except:
            x = row[[str(dim) for dim in range(self.num_dims)]].values


        sample_idx = row["sample_idx"]
        dataset_idx = row["dataset_idx"]

        # make batch encoding
        one_hot_modality = np.ones((1,), dtype=np.float32) # TODO: change this to be more general from ones to one hot
        one_hot_sample = np.zeros((self.num_samples,), dtype=np.float32)
        one_hot_study = np.zeros((self.num_studies,), dtype=np.float32)
        if not self.predict_mode:
            one_hot_sample[sample_idx] = 1
            one_hot_study[dataset_idx] = 1

        # library
        if sample_idx in self.library_calcs.index:
            local_l_mean = self.library_calcs.loc[sample_idx, "library_log_means"]
            local_l_var = self.library_calcs.loc[sample_idx, "library_log_vars"]
        else:
            local_l_mean = 0.0
            local_l_var = 1.0
        
        # make return
        ret = {
            "x": torch.from_numpy(x.astype("float32")),
            "modality_vec": torch.from_numpy(one_hot_modality),
            "study_vec": torch.from_numpy(one_hot_study),
            "sample_vec": torch.from_numpy(one_hot_sample),
            "local_l_mean": torch.tensor(local_l_mean),
            "local_l_var": torch.tensor(local_l_var),
            "feature_presence_mask": torch.from_numpy(np.ones(self.num_dims, dtype=np.float32)),
            "soma_joinid": torch.tensor(soma_joinid, dtype=torch.float32),
            "manual_index": torch.tensor(manual_index)
        }

        return ret
            



