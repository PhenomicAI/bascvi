from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import IterableDataset
import math
from .soma_helpers import open_soma_experiment

import tiledbsoma as soma



class TileDBSomaTorchIterDataset(IterableDataset):
    """Custom torch dataset to get data from tiledbsoma in tensor form for pytorch modules."""
       
    def __init__(
        self,
        soma_experiment_uri,
        obs_df,
        num_input,
        genes_to_use,
        feature_presence_matrix,
        block_size,
        num_workers,
        num_modalities=None,
        num_studies=None,
        num_samples=None,
        library_calcs=None,
        verbose=False,
        predict_mode=False,
        pretrained_gene_indices = None
    ):     
        self.soma_experiment_uri = soma_experiment_uri

        self.obs_df = obs_df
        self.genes_to_use = genes_to_use

        self.X_array_name = "row_raw"
        self.feature_presence_matrix = feature_presence_matrix

        self.predict_mode = predict_mode

        self.num_input = num_input

        self.num_modalities = num_modalities
        self.num_studies = num_studies
        self.num_samples = num_samples

        if self.predict_mode:
            assert self.num_modalities is None
            assert self.num_studies is None
            assert self.num_samples is None


        self.block_counter = 0
        self.cell_counter = 0

        self.block_size = block_size
        self.num_blocks = math.ceil(self.obs_df.shape[0] / self.block_size) 

        self.library_calcs = library_calcs

        self.num_workers = num_workers

        self._len = self.obs_df.shape[0]

        self.verbose = verbose

        self.pretrained_gene_indices = pretrained_gene_indices

        assert self.obs_df.soma_joinid.nunique() == self.obs_df.shape[0]
        assert self.obs_df.cell_idx.nunique() == self.obs_df.shape[0]

    def __len__(self):
        return self._len
    
    def _calc_start_end(self, worker_id):

        # we have less blocks than workers
        if self.num_blocks < self.num_workers:
            # change num_blocks and block_size
            self.num_blocks = self.num_workers
            self.block_size = math.ceil(self.obs_df.shape[0] / self.num_blocks)

            start_block = worker_id
            end_block = worker_id + 1

        else:
            num_blocks_per_worker = math.floor(self.num_blocks / self.num_workers)
            start_block = worker_id * num_blocks_per_worker
            end_block = start_block + num_blocks_per_worker

            if worker_id + 1 == self.num_workers:
                end_block = self.num_blocks

        return (start_block, end_block)
    
    def __iter__(self):
        if torch.utils.data.get_worker_info():
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id

            self.start_block, self.end_block = self._calc_start_end(self.worker_id)
        else:
            self.start_block = 0
            self.end_block = self.num_blocks

        # Reset counters at the start of each epoch (this is for persistent workers)
        self.block_counter = 0
        self.cell_counter = 0

        if self.verbose:
            print("start block, end block, block_size: ", self.start_block, self.end_block, self.block_size)
        return self

    def __next__(self):
        if self.verbose:
            print("begin next loop...")

        self.curr_block = self.start_block + self.block_counter 
        
        if self.curr_block < self.end_block:

            if self.cell_counter == 0:

                # get block start and end
                start_idx = self.curr_block * self.block_size
                end_idx = min(start_idx + self.block_size, self.obs_df.shape[0])  


                if self.verbose:
                    print("reading new block of size ", end_idx - start_idx)

                # get block data
                self.obs_df_block = self.obs_df.iloc[start_idx:end_idx, : ]

                self.cell_idx_block = self.obs_df_block['cell_idx'].to_numpy(dtype=np.int64)
                self.soma_joinid_block = self.obs_df_block["soma_joinid"].to_numpy(dtype=np.int64)

                self.modality_idx_block = self.obs_df_block["modality_idx"].to_numpy()
                self.study_idx_block = self.obs_df_block["study_idx"].to_numpy()
                self.sample_idx_block = self.obs_df_block["sample_idx"].to_numpy()

                assert self.obs_df_block.shape[0] == (end_idx - start_idx)
                assert len(np.unique(self.soma_joinid_block)) == (end_idx - start_idx)
                assert len(np.unique(self.cell_idx_block)) == (end_idx - start_idx)

                if self.verbose:
                    print("num cells in block: ", (end_idx - start_idx))
                # if not self.soma_joinid_block:
                #     print(self.soma_joinid_block)
                #     print(self.num_cells_in_block, self.start_block, self.curr_block, self.end_block, self.num_blocks)
                #     return
                try:
                    # read block
                    with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                        # OLD WAY
                        # self.X_block = soma_experiment.ms["RNA"]["X"][self.X_array_name].read((tuple(self.soma_joinid_block), None)).coos(shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[self.soma_joinid_block, :]
                        with soma_experiment.axis_query("RNA", obs_query=soma.AxisQuery(coords=(tuple(self.soma_joinid_block),))) as query:
                            adata = query.to_anndata(X_name='row_raw', column_names={"obs":["soma_joinid"], "var":[]})
                            adata.obs_names = adata.obs["soma_joinid"].astype(str)

                    # make soma_joinid_block a list of strings
                    soma_joinid_block_str = [str(x) for x in self.soma_joinid_block]
                    adata = adata[soma_joinid_block_str, :]

                    assert np.all(adata.obs["soma_joinid"] == self.soma_joinid_block)

                    self.X_block = adata.X
                    
                    self.X_block = self.X_block[:, self.genes_to_use]
                    


                except Exception as error:
                    print("Error reading X array of block: ", self.curr_block)
                    print(error)
                    raise ValueError()
                
            if self.verbose:    
                print("subsetting, converting, and transposing x...")

            # set X_curr and sample_idx_curr
            X_curr = np.squeeze(np.transpose(self.X_block[self.cell_counter, :].toarray()))
            if self.pretrained_gene_indices is not None:
                # expand X_curr to full size of pretrained model
                X_curr_full = np.zeros(self.num_input,  dtype=np.int32)
                X_curr_full[self.pretrained_gene_indices] = X_curr
                X_curr = np.squeeze(np.transpose(X_curr_full))


            sample_idx_curr = self.sample_idx_block[self.cell_counter]

            soma_joinid = self.soma_joinid_block[self.cell_counter]
            cell_idx = self.cell_idx_block[self.cell_counter]
            feature_presence_mask = self.feature_presence_matrix[sample_idx_curr, :]


            if self.predict_mode:
                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
                    "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
                    "feature_presence_mask": torch.from_numpy(feature_presence_mask),                
                }

            else: # training / validation mode
                modality_idx_curr = self.modality_idx_block[self.cell_counter]
                study_idx_curr = self.study_idx_block[self.cell_counter]

                # construct batch vecs
                one_hot_modality = np.zeros((self.num_modalities,), dtype=np.float32)
                one_hot_study = np.zeros((self.num_studies,), dtype=np.float32)
                one_hot_sample = np.zeros((self.num_samples,), dtype=np.float32)

                one_hot_modality[modality_idx_curr] = 1
                one_hot_study[study_idx_curr] = 1
                one_hot_sample[sample_idx_curr] = 1
            
                # library
                if sample_idx_curr in self.library_calcs.index:
                    local_l_mean = self.library_calcs.loc[sample_idx_curr, "library_log_means"]
                    local_l_var = self.library_calcs.loc[sample_idx_curr, "library_log_vars"]
                else:
                    local_l_mean = 0.0
                    local_l_var = 1.0

                # make return
                datum = {
                    "x": torch.from_numpy(X_curr.astype("int32")),
                    "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
                    "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
                    "feature_presence_mask": torch.from_numpy(feature_presence_mask),  
                    "modality_vec": torch.from_numpy(one_hot_modality),
                    "study_vec": torch.from_numpy(one_hot_study),
                    "sample_vec": torch.from_numpy(one_hot_sample),
                    "local_l_mean": torch.tensor(local_l_mean),
                    "local_l_var": torch.tensor(local_l_var),
                }

            # increment counters
            if (self.cell_counter + 1) == self.obs_df_block.shape[0]:
                self.block_counter += 1
                self.cell_counter = 0
            else:
                self.cell_counter += 1

            return datum
        else:
            raise StopIteration
            



