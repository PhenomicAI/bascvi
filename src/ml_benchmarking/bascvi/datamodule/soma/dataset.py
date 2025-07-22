import math
from typing import Dict, List, Optional, Any, Iterator

import torch
import numpy as np
import tiledbsoma as soma
from torch.utils.data import IterableDataset

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment

class TileDBSomaTorchIterDataset(IterableDataset):
    """
    Custom torch IterableDataset to fetch data from TileDB-SOMA in tensor form for PyTorch modules.
    Supports multi-worker iteration and block-wise data loading.
    """
    
    def __init__(
        self,
        soma_experiment_uri: str,
        obs_df: Any,  # Typically a pandas DataFrame
        num_input: int,
        genes_to_use: np.ndarray,
        feature_presence_matrix: np.ndarray,
        block_size: int,
        num_workers: int,
        num_modalities: Optional[int] = None,
        num_studies: Optional[int] = None,
        num_samples: Optional[int] = None,
        library_calcs: Optional[Any] = None,  # Typically a pandas DataFrame
        verbose: bool = False,
        predict_mode: bool = False,
        pretrained_gene_indices: Optional[np.ndarray] = None
    ) -> None:
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
        self.library_calcs = library_calcs
        self.num_workers = num_workers
        self._len = self.obs_df.shape[0]
        self.verbose = verbose
        self.pretrained_gene_indices = pretrained_gene_indices

        #Cache info on obs_df
        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            self.rna_X = soma_experiment.ms["RNA"]["X"][self.X_array_name]
            self.full_shape = (
                soma_experiment.obs.count,
                soma_experiment.ms["RNA"].var.count
            )
    
        # Validate input DataFrame
        assert self.obs_df.soma_joinid.nunique() == self.obs_df.shape[0], "soma_joinid must be unique per row."
        assert self.obs_df.cell_idx.nunique() == self.obs_df.shape[0], "cell_idx must be unique per row."
        if self.predict_mode:
            assert self.num_modalities is None
            assert self.num_studies is None
            assert self.num_samples is None

        # Block and cell counters for iteration
        self.block_counter = 0
        self.cell_counter = 0
        self.block_size = block_size
        self.num_blocks = math.ceil(self.obs_df.shape[0] / self.block_size)

    def __len__(self) -> int:
        return self._len
    
    def _calc_start_end(self, worker_id: int) -> tuple:
        if self.num_blocks < self.num_workers:
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            self.worker_id = worker_info.id
            self.start_block, self.end_block = self._calc_start_end(self.worker_id)
        else:
            self.start_block = 0
            self.end_block = self.num_blocks
        self.block_counter = 0
        self.cell_counter = 0
        if self.verbose:
            print("start block, end block, block_size: ", self.start_block, self.end_block, self.block_size)
        return self

    def _get_block(self, curr_block):
        start_idx = curr_block * self.block_size
        end_idx = min(start_idx + self.block_size, self.obs_df.shape[0])
        if self.verbose:
            print("reading new block of size ", end_idx - start_idx)
        obs_df_block = self.obs_df.iloc[start_idx:end_idx, : ]
        cell_idx_block = obs_df_block['cell_idx'].to_numpy(dtype=np.int64)
        soma_joinid_block = obs_df_block["soma_joinid"].to_numpy(dtype=np.int64)
        modality_idx_block = obs_df_block["modality_idx"].to_numpy()
        study_idx_block = obs_df_block["study_idx"].to_numpy()
        sample_idx_block = obs_df_block["sample_idx"].to_numpy()
        assert obs_df_block.shape[0] == (end_idx - start_idx)
        assert len(np.unique(soma_joinid_block)) == (end_idx - start_idx)
        assert len(np.unique(cell_idx_block)) == (end_idx - start_idx)
        try:
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                sorted_soma_joinids = sorted(soma_joinid_block)

                sparse_matrix = (
                    self.rna_X
                    .read((tuple(sorted_soma_joinids), None))
                    .coos(shape=self.full_shape)
                    .concat()
                    .to_scipy()
                    .tocsr()
                )

                X_block = sparse_matrix[soma_joinid_block, :]
                nan_mask = np.isnan(X_block.data)
                X_block.data[nan_mask] = 0

            X_block = X_block[:, self.genes_to_use]
        except Exception as error:
            print("Error reading X array of block: ", curr_block)
            print(error)
            raise ValueError()
        return (obs_df_block, cell_idx_block, soma_joinid_block, modality_idx_block, study_idx_block, sample_idx_block, X_block)

    def _make_datum(self, X_curr, soma_joinid, cell_idx, feature_presence_mask, sample_idx_curr, modality_idx_curr=None, study_idx_curr=None):
        base = {
            "x": torch.from_numpy(X_curr.astype("int32")),
            "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
            "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
            "feature_presence_mask": torch.from_numpy(feature_presence_mask),
        }
        if self.predict_mode:
            return base
        # Add training/validation fields
        one_hot_modality = np.zeros((self.num_modalities,), dtype=np.float32)
        one_hot_study = np.zeros((self.num_studies,), dtype=np.float32)
        one_hot_sample = np.zeros((self.num_samples,), dtype=np.float32)
        one_hot_modality[modality_idx_curr] = 1
        one_hot_study[study_idx_curr] = 1
        one_hot_sample[sample_idx_curr] = 1
        if sample_idx_curr in self.library_calcs.index:
            local_l_mean = self.library_calcs.loc[sample_idx_curr, "library_log_means"]
            local_l_var = self.library_calcs.loc[sample_idx_curr, "library_log_vars"]
        else:
            local_l_mean = 0.0
            local_l_var = 1.0
        base.update({
            "modality_vec": torch.from_numpy(one_hot_modality),
            "study_vec": torch.from_numpy(one_hot_study),
            "sample_vec": torch.from_numpy(one_hot_sample),
            "local_l_mean": torch.tensor(local_l_mean),
            "local_l_var": torch.tensor(local_l_var),
        })
        return base

    def __next__(self) -> Dict[str, Any]:
        if self.verbose:
            print("begin next loop...")
        self.curr_block = self.start_block + self.block_counter 
        if self.curr_block < self.end_block:
            if self.cell_counter == 0:
                (
                    self.obs_df_block,
                    self.cell_idx_block,
                    self.soma_joinid_block,
                    self.modality_idx_block,
                    self.study_idx_block,
                    self.sample_idx_block,
                    self.X_block
                ) = self._get_block(self.curr_block)
            if self.verbose:    
                print("subsetting, converting, and transposing x...")
            X_curr = np.squeeze(np.transpose(self.X_block[self.cell_counter, :].toarray()))
            if self.pretrained_gene_indices is not None:
                X_curr_full = np.zeros(self.num_input,  dtype=np.int32)
                X_curr_full[self.pretrained_gene_indices] = X_curr
                X_curr = np.squeeze(np.transpose(X_curr_full))
            sample_idx_curr = self.sample_idx_block[self.cell_counter]
            soma_joinid = self.soma_joinid_block[self.cell_counter]
            cell_idx = self.cell_idx_block[self.cell_counter]
            feature_presence_mask = self.feature_presence_matrix[sample_idx_curr, :]
            if self.predict_mode:
                datum = self._make_datum(X_curr, soma_joinid, cell_idx, feature_presence_mask, sample_idx_curr)
            else:
                modality_idx_curr = self.modality_idx_block[self.cell_counter]
                study_idx_curr = self.study_idx_block[self.cell_counter]
                datum = self._make_datum(
                    X_curr, soma_joinid, cell_idx, feature_presence_mask, sample_idx_curr, modality_idx_curr, study_idx_curr
                )
            if (self.cell_counter + 1) == self.obs_df_block.shape[0]:
                self.block_counter += 1
                self.cell_counter = 0
            else:
                self.cell_counter += 1
            return datum
        else:
            raise StopIteration
            



