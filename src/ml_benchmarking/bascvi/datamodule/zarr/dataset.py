import math
from typing import Dict, List
import anndata
import numpy as np
import torch
from torch.utils.data import IterableDataset
import gc

class ZarrDataset(IterableDataset):
    """Custom torch dataset to get data from zarr in tensor form for pytorch modules."""
       
    def __init__(
        self,
        file_paths,
        reference_gene_list,
        zarr_len_dict,
        num_batches,
        num_workers,
        predict_mode=False
    ):

        self.reference_gene_list = reference_gene_list

        # Create a dummy AnnData for reference gene alignment
        self.ref_adata = anndata.AnnData(
            X=np.zeros((1, len(self.reference_gene_list)), dtype=np.float32),
            var={'gene': self.reference_gene_list},
            dtype=np.float32
        )
        self.ref_adata.var = self.ref_adata.var.set_index(self.ref_adata.var['gene'])

        self.num_files = len(file_paths)
        self.file_paths = file_paths

        self.num_workers = num_workers
        self.num_batches = num_batches

        self.predict_mode = predict_mode

        self._len = 0
        for p in file_paths:
            self._len += zarr_len_dict[p]

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
                # Only support zarr
                self.adata = anndata.read_zarr(curr_adata_path)

                # ensure genes are lower case
                self.adata.var['gene'] = self.adata.var['gene'].str.lower()
                self.adata.var = self.adata.var.set_index(self.adata.var['gene'])

                self.adata.X = self.adata.X.astype(np.int32)

                # make index for locating cell in adata in case we subset
                self.adata.obs['int_index'] = list(range(self.adata.shape[0]))


                # should be Prod specific (prod means predicting on new data)
                #if self.prod_mode:

                self.feature_presence_mask = np.asarray([r in self.adata.var['gene'] for r in self.reference_gene_list])
                print('# genes found in reference: ', np.sum(self.feature_presence_mask), '# genes in adata', self.adata.shape[1], '# genes in reference', len(self.reference_gene_list))

                # expands the adata to include the reference genes
                self.adata = anndata.concat([self.ref_adata.copy(), self.adata], join='outer')
                # remove the empty top row from ref, subset genes to reference gene list
                self.adata = self.adata[1:, self.reference_gene_list]

                self.adata.obs = self.adata.obs.set_index(self.adata.obs['int_index'], drop=False)
                self.adata.var = self.adata.var.reset_index()

                # training mode      
                if not self.predict_mode:
                    # filter cells with very few gene reads (but be more lenient for small datasets)
                    gene_counts = np.sum(self.adata.X > 0, axis=1)
                    min_genes = min(300, self.adata.shape[0] // 2)  # Be more lenient for small datasets
                    mask = gene_counts > min_genes
                    self.adata = self.adata[mask, :]
                    
                    # Check if we still have cells after filtering
                    if self.adata.shape[0] == 0:
                        # If no cells pass filter, use all cells
                        self.adata = anndata.read_zarr(curr_adata_path)
                        self.adata.var['gene'] = self.adata.var['gene'].str.lower()
                        self.adata.var = self.adata.var.set_index(self.adata.var['gene'])
                        self.adata = anndata.concat([self.ref_adata.copy(), self.adata], join='outer')
                        self.adata = self.adata[1:, self.reference_gene_list]

                    # make batch vector - using file index as batch
                    one_hot_batch = np.zeros(3)  # [modality, study, sample]
                    one_hot_batch[2] = self.file_counter  # sample index

                    # get library calcs
                    self.adata.obs['int_index'] = list(range(self.adata.shape[0]))
                    # For zarr files, we'll use simple library calculations
                    total_counts = np.sum(self.adata.X, axis=1)
                    log_counts = np.log(total_counts + 1)
                    self.l_mean_all = np.mean(log_counts)
                    self.l_var_all = np.var(log_counts)

            # check if adata.X is a sparse matrix
            if isinstance(self.adata.X, np.ndarray): # Changed from csr_matrix to np.ndarray for zarr
                X_curr = np.squeeze(self.adata.X[self.cell_counter, :])
            else: # This case should ideally not be reached for zarr
                X_curr = np.squeeze(self.adata.X[self.cell_counter, :])
            

            if self.predict_mode:
                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    "locate": torch.tensor([self.file_counter, self.adata.obs['int_index'].values.tolist()[self.cell_counter]]),
                    "feature_presence_mask": torch.from_numpy(self.feature_presence_mask),                
                }

            else:
                
                local_l_mean = self.l_mean_all
                local_l_var = self.l_var_all

                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    "batch_emb": torch.from_numpy(one_hot_batch),
                    "local_l_mean": torch.tensor(local_l_mean),
                    "local_l_var": torch.tensor(local_l_var),
                    "feature_presence_mask": torch.from_numpy(self.feature_presence_mask),
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

