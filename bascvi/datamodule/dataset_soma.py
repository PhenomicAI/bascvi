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



class TileDBSomaTorchDataset(Dataset):
    """Custom torch dataset to get data from tiledbsoma in tensor form for pytorch modules."""
       
    def __init__(
        self,
        soma_experiment,
        samples_list,
        num_total_batches,
        num_genes,
        workers,
        l_means,
        l_vars,
        pred_mode=False
    ):     
        self.soma_experiment = soma_experiment
        self.X_row_norm = self.soma_experiment.ms["RNA"]["X"]["row_norm"]
        self.feature_presence_matrix = self.soma_experiment.ms["RNA"]["feature_presence_matrix"].read(coords=(tuple(samples_list), None)).coos(shape=(num_total_batches, num_genes)).concat().to_scipy().toarray()

        self.samples_list = samples_list

        self.pred_mode = pred_mode

        self.num_total_batches = num_total_batches

        self.sample_counter = 0
        self.cell_counter = 0

        self.l_means = l_means
        self.l_vars = l_vars

        self.num_cells = self.soma_experiment.obs.count
        self.num_genes = num_genes

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # with self.soma_experiment.axis_query(measurement_name="RNA", obs_query=soma.AxisQuery(coords=((idx, )))) as query:
        #     adata: sc.AnnData = query.to_anndata(
        #         X_name="row_norm",
        #         column_names={"obs": ["soma_joinid", "sample_idx"], "var": ["soma_joinid"]},
        #     )
        #     X = np.squeeze(np.transpose(adata[:, :].X.toarray()))
        #     sample_idx = int(adata.obs["sample_idx"][0])
        
        X = np.squeeze(np.transpose(self.X_row_norm.read((idx,)).coos(shape=(self.num_cells, self.num_genes)).concat().to_scipy().tocsr()[idx, :].toarray()))
        sample_idx = int(self.soma_experiment.obs.read((idx,), column_names=["soma_joinid", "sample_idx"]).concat().to_pandas()["sample_idx"][0])

        
        # make batch encoding
        one_hot = np.zeros((self.num_total_batches,),dtype=np.float32)
        if not self.pred_mode:
            one_hot[sample_idx] = 1



        return {
                "x": torch.from_numpy(X),
                "batch_emb": torch.from_numpy(one_hot),
                "local_l_mean": torch.tensor(self.l_means[sample_idx]),
                "local_l_var": torch.tensor(self.l_vars[sample_idx]),
                "mask": torch.from_numpy(self.feature_presence_matrix[sample_idx, :])
                }
            



