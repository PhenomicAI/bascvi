import math
from typing import Dict, List
import anndata
import scanpy
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import IterableDataset
import gc
import pickle as pkl
import pandas as pd
import os
from fast_matrix_market import mmread
from scipy.sparse import csr_matrix
import time

class AnnDataDataset(IterableDataset):
    """Custom torch dataset to get data from anndata in tensor form for pytorch modules."""
       
    def __init__(
        self,
        file_paths,
        reference_gene_list,
        adata_len_dict,
        num_batches,
        num_workers,
        predict_mode=False,
        batch_mappings=None,
        batch_sizes=None,
        batch_keys=None
    ):
        self.reference_gene_list = reference_gene_list
        self.file_paths = file_paths
        self.adata_len_dict = adata_len_dict
        self.num_batches = num_batches
        self.num_workers = num_workers
        self.predict_mode = predict_mode
        self.batch_mappings = batch_mappings
        self.batch_sizes = batch_sizes
        self.batch_keys = batch_keys

        # Create reference adata for gene alignment
        self.ref_adata = scanpy.AnnData(
            X=np.zeros((1, len(self.reference_gene_list)), dtype=np.float32), 
            var={'gene': self.reference_gene_list},
            dtype=np.float32
        )
        self.ref_adata.var = self.ref_adata.var.set_index(self.ref_adata.var['gene'])

        self.num_files = len(file_paths)
        self._len = 0
        for p in file_paths:
            self._len += adata_len_dict[p]

        self.file_counter = 0
        self.cell_counter = 0

    def __len__(self):
        return self._len

    def __iter__(self):
        if torch.utils.data.get_worker_info():
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id
            self.start_file, self.end_file = self._calc_start_end(self.worker_id)
        else:
            self.start_file = 0
            self.end_file = self.num_files

        return self

    def _calc_start_end(self, worker_id):
        # we have less or equal num files to workers
        if self.num_files <= self.num_workers:
            # assign one file per worker
            self.num_files_per_worker = 1
            start_file = worker_id
            end_file = worker_id + 1
        # we have more files than workers
        else:
            # calculate number of files per worker
            self.num_files_per_worker = math.floor(self.num_files / self.num_workers)
            start_file = worker_id * self.num_files_per_worker
            end_file = start_file + self.num_files_per_worker

            if worker_id + 1 == self.num_workers:
                end_file = self.num_files

        return (start_file, end_file)
    
    def __next__(self):
        if self.file_counter + self.start_file < self.end_file:
            curr_adata_path = self.file_paths[self.file_counter + self.start_file]

            # if count is 0 then we need to load a new adata from disk
            if self.cell_counter == 0: 
                # deal with different file types
                if curr_adata_path[-6:] == 'mtx.gz':
                    mtx_ = mmread(curr_adata_path).astype(np.float32)
                    obs_ = pd.read_csv(curr_adata_path[:-13] + 'barcodes.tsv.gz', delimiter='\t', header=None)
                    var_ = pd.read_csv(curr_adata_path[:-13] + 'features.tsv.gz', delimiter='\t', header=None)
                    
                    var_['gene'] = var_[0]
                    var_ = var_.set_index(var_[0])
                    var_ = var_.iloc[:,-1:]

                    self.adata = scanpy.AnnData(X=csr_matrix(mtx_.T),obs=obs_, var=var_)
                    self.adata.obs['sample_name'] = self.file_counter + self.start_file

                else:
                    self.adata = scanpy.read(curr_adata_path)

                # ensure genes are lower case
                self.adata.var['gene'] = self.adata.var['gene'].str.lower()
                self.adata.var = self.adata.var.set_index(self.adata.var['gene'])

                self.adata.X = self.adata.X.astype(np.int32)

                # make index for locating cell in adata in case we subset
                self.adata.obs['int_index'] = list(range(self.adata.shape[0]))

                self.feature_presence_mask = np.asarray([r in self.adata.var['gene'] for r in self.reference_gene_list])
                print('# genes found in reference: ', np.sum(self.feature_presence_mask), '# genes in adata', self.adata.shape[1], '# genes in reference', len(self.reference_gene_list))

                # expands the adata to include the reference genes
                self.adata = scanpy.concat([self.ref_adata.copy(), self.adata], join='outer')
                # remove the empty top row from ref, subset genes to reference gene list
                self.adata = self.adata[1:, self.reference_gene_list]

                self.adata.obs = self.adata.obs.set_index(self.adata.obs['int_index'], drop=False)
                self.adata.var = self.adata.var.reset_index()

                # training mode      
                if not self.predict_mode:
                    # filter cells with very few gene reads
                    gene_counts = self.adata.X.getnnz(axis=1)
                    mask = gene_counts > 300
                    self.adata = self.adata[mask, :]

                    # get library calcs
                    self.adata.obs['int_index'] = list(range(self.adata.shape[0]))
                    self.l_mean_all = self.adata.obs.groupby("sample_name")["int_index"].transform(log_mean, self.adata.X)
                    self.l_var_all = self.adata.obs.groupby("sample_name")["int_index"].transform(log_var, self.adata.X)

            # check if adata.X is a sparse matrix
            if isinstance(self.adata.X, csr_matrix):
                X_curr = np.squeeze(self.adata.X[self.cell_counter, :].toarray())
            else:
                X_curr = np.squeeze(self.adata.X[self.cell_counter, :])

            if self.predict_mode:
                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    "locate": torch.tensor([self.file_counter, self.adata.obs['int_index'].values.tolist()[self.cell_counter]]),
                    "feature_presence_mask": torch.from_numpy(self.feature_presence_mask),                
                }
            else:
                # Create one-hot vectors for each batch level
                batch_vectors = {}
                for level, col in self.batch_keys.items():
                    cat = self.adata.obs[col].values[self.cell_counter]
                    idx = self.batch_mappings[level][cat]
                    one_hot = np.zeros(self.batch_sizes[level], dtype=np.float32)
                    one_hot[idx] = 1
                    batch_vectors[level] = torch.from_numpy(one_hot)

                local_l_mean = self.l_mean_all[self.cell_counter]
                local_l_var = self.l_var_all[self.cell_counter]

                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    **{f"{level}_vec": vec for level, vec in batch_vectors.items()},
                    "local_l_mean": torch.tensor(local_l_mean),
                    "local_l_var": torch.tensor(local_l_var),
                    "feature_presence_mask": torch.tensor(self.feature_presence_mask),
                    "locate": torch.tensor([self.file_counter, self.adata.obs['int_index'].values.tolist()[self.cell_counter]])
                }

            # if done with all cells in adata, set cell counter to 0 and move on to next file
            if self.cell_counter + 1 == self.adata.shape[0]:
                del self.adata
                gc.collect()

                self.cell_counter = 0
                self.file_counter += 1     
            else:
                self.cell_counter += 1   
             
            return datum

        else:
            raise StopIteration

def log_mean(g, X):
    vals = X[g.values,:]
    log_counts = np.log(vals.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(g, X):
    vals = X[g.values,:]
    log_counts = np.log(vals.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var

