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

class AnnDataDataset(IterableDataset):
    """Custom torch dataset to get data from anndata in tensor form for pytorch modules."""
       
    def __init__(
        self,
        file_paths,
        reference_gene_list,
        adata_len_dict,
        num_batches,
        num_workers,
        batch_dict=None,
        predict_mode=False
    ):

        self.reference_gene_list = reference_gene_list

        self.ref_adata = scanpy.AnnData(
            X=np.zeros((1, len(self.reference_gene_list)), dtype=np.float32), 
            var={'gene': self.reference_gene_list}
            )
        self.ref_adata.var = self.ref_adata.var.set_index(self.ref_adata.var['gene'])

        self.batch_dict = batch_dict
        self.num_files = len(file_paths)
        self.file_paths = file_paths

        self.num_workers = num_workers
        self.num_batches = num_batches

        self.predict_mode = predict_mode

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

        # print("start block, end block, block_size: ", self.start_block, self.end_block, self.block_size)

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

                print(self.adata.X.shape)

                # make index for locating cell in adata in case we subset
                self.adata.obs['int_index'] = list(range(self.adata.shape[0]))

                # should be Prod specific (prod means predicting on new data)
                #if self.prod_mode:

                self.feature_presence_mask = np.asarray([r in self.adata.var['gene'] for r in self.reference_gene_list])
                print('# genes found in reference: ', np.sum(self.feature_presence_mask), '# genes in adata', self.adata.shape[1], '# genes in reference', len(self.reference_gene_list))

                # expands the adata to include the reference genes
                self.adata = scanpy.concat([self.ref.copy(), self.adata], join='outer')
                # remove the empty top row from ref, subset genes to reference gene list
                self.adata = self.adata[1:, self.reference_gene_list]

                # predict mode
                if self.predict_mode:

                    # dummy batch vec
                    one_hot_batch = np.zeros((self.num_batches,), dtype=np.float32)

                    # dummy library calcs
                    self.l_mean_all = np.zeros(self.adata.shape[0], dtype=np.float32)
                    self.l_var_all = np.ones(self.adata.shape[0], dtype=np.float32)

                # training mode      
                else:

                    raise("Training mode for adata not implemented yet")
                
                    # filter cells with very few gene reads
                    gene_counts = self.adata.X.getnnz(axis=1)
                    mask = gene_counts > 300
                    self.adata = self.adata[mask, :]

                    # make batch vector
                    # TODO: make this work for training
                    one_hot_batch = ...

                    # get library calcs
                    self.adata.obs['int_index'] = list(range(self.adata.shape[0]))
                    self.l_mean_all = self.adata.obs.groupby("sample_name")["int_index"].transform(log_mean, self.adata.X)
                    self.l_var_all = self.adata.obs.groupby("sample_name")["int_index"].transform(log_var, self.adata.X)

                          
            local_l_mean = self.l_mean_all[self.cell_counter]
            local_l_var = self.l_var_all[self.cell_counter]

            x = np.squeeze(self.adata.X[self.cell_counter].copy().toarray())
            
            # make return
            datum = {
                "x": torch.from_numpy(x),
                "batch_emb": torch.from_numpy(one_hot_batch),
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
                self.cell_counter      
             
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

