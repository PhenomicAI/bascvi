import math
from typing import Dict, List, Optional, Any, Iterator
import torch
import numpy as np
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import anndata as ad
import os
from glob import glob
import zarr

class ZarrDataset(IterableDataset):
    """
    IterableDataset for loading AnnData zarr files for PyTorch models.
    Each datum contains: x, cell_idx, feature_presence_mask, modality_vec, study_vec, sample_vec, local_l_mean, local_l_var.
    """
    def __init__(
        self,
        data_root_dir: str,
        feature_presence_matrix: np.ndarray,
        batch_level_sizes: list,
        block_size: int = 64,
        num_workers: int = 1,
        verbose: bool = False,
        predict_mode: bool = False,
        pretrained_gene_indices: Optional[np.ndarray] = None,
        gene_list: Optional[List[str]] = None,
        validation_split: float = 0.1,
        is_validation: bool = False,
    ) -> None:

        self.data_root_dir = data_root_dir
        self.feature_presence_matrix = feature_presence_matrix
        self.predict_mode = predict_mode
        self.num_workers = num_workers
        self.verbose = verbose
        self.pretrained_gene_indices = pretrained_gene_indices
        self.block_size = block_size
        self.gene_list = gene_list
        self.num_modalities, self.num_studies, self.num_samples = batch_level_sizes
        self.validation_split = validation_split
        self.is_validation = is_validation

        self.zarr_files = sorted(glob(os.path.join(self.data_root_dir, '*.zarr')))
        assert self.zarr_files, f"No .zarr files found in {self.data_root_dir}"
        self.block_cell_counts = [zarr.open_group(zf, mode="r")['X'].attrs['shape'][0] for zf in self.zarr_files]
        self._len = sum(self.block_cell_counts)
        self.num_blocks = len(self.zarr_files)
        self._block_data = None
        self.block_counter = 0
        self.cell_counter = 0

    def __len__(self) -> int:
        # Calculate length based on the split
        if self.is_validation:
            # Use last validation_split portion of blocks
            split_point = int(self.num_blocks * (1 - self.validation_split))
            validation_blocks = self.block_cell_counts[split_point:]
            return sum(validation_blocks)
        else:
            # Use first (1 - validation_split) portion of blocks
            split_point = int(self.num_blocks * (1 - self.validation_split))
            training_blocks = self.block_cell_counts[:split_point]
            return sum(training_blocks)

    def _calc_start_end(self, worker_id: int) -> tuple:
        # First, determine the overall split
        if self.is_validation:
            # Use last validation_split portion of blocks
            split_point = int(self.num_blocks * (1 - self.validation_split))
            overall_start = split_point
            overall_end = self.num_blocks
            if worker_id == 0:  # Only print once per dataset
                print(f"Validation split: blocks {overall_start} to {overall_end} (total blocks: {self.num_blocks})")
        else:
            # Use first (1 - validation_split) portion of blocks
            split_point = int(self.num_blocks * (1 - self.validation_split))
            overall_start = 0
            overall_end = split_point
            if worker_id == 0:  # Only print once per dataset
                print(f"Training split: blocks {overall_start} to {overall_end} (total blocks: {self.num_blocks})")
        
        # Now assign workers within the appropriate range
        available_blocks = overall_end - overall_start
        if available_blocks < self.num_workers:
            # More workers than blocks, assign one block per worker
            if worker_id < available_blocks:
                start_block = overall_start + worker_id
                end_block = start_block + 1
            else:
                # This worker gets no blocks
                start_block = overall_end
                end_block = overall_end
        else:
            # Distribute blocks among workers
            blocks_per_worker = math.floor(available_blocks / self.num_workers)
            start_block = overall_start + (worker_id * blocks_per_worker)
            end_block = start_block + blocks_per_worker
            if worker_id + 1 == self.num_workers:
                end_block = overall_end
        
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
        self._block_data = None
        return self

    def _get_block(self, curr_block):
        zarr_path = self.zarr_files[curr_block]
        adata = ad.read_zarr(zarr_path)
        obs_df_block = adata.obs.reset_index()
        X_block = adata.X
        # Ensure cell_idx is properly converted to int64 and handle any potential overflow
        cell_idx_block = obs_df_block["cell_idx"].astype(np.int64).to_numpy()
        modality_idx_block = np.zeros(obs_df_block.shape[0], dtype=np.int64)
        study_idx_block = obs_df_block["study_idx"].to_numpy()
        sample_idx_block = obs_df_block["sample_idx"].to_numpy()
        local_l_mean_block = obs_df_block["log_mean"].to_numpy()
        local_l_var_block = obs_df_block["log_var"].to_numpy()

        return (cell_idx_block, modality_idx_block, study_idx_block, sample_idx_block, X_block, local_l_mean_block, local_l_var_block)

    def _make_datum(self, X_curr, cell_idx, feature_presence_mask, modality_idx, study_idx, sample_idx, local_l_mean, local_l_var):
        base = {
            "x": torch.from_numpy(X_curr.astype("int32")),
            "cell_idx": torch.tensor(int(cell_idx), dtype=torch.int64),
            "feature_presence_mask": torch.from_numpy(feature_presence_mask),
            "modality_vec": F.one_hot(torch.tensor(modality_idx, dtype=torch.long), num_classes=self.num_modalities).float(),
            "study_vec": F.one_hot(torch.tensor(study_idx, dtype=torch.long), num_classes=self.num_studies).float(),
            "sample_vec": F.one_hot(torch.tensor(sample_idx, dtype=torch.long), num_classes=self.num_samples).float(),
            "local_l_mean": torch.tensor(0.0 if np.isinf(local_l_mean) else local_l_mean),
            "local_l_var": torch.tensor(1.0 if np.isnan(local_l_var) else local_l_var),
        }
        return base

    def __next__(self) -> Dict[str, Any]:
        self.curr_block = self.start_block + self.block_counter
        if self.curr_block < self.end_block:
            if self.cell_counter == 0:
                self._block_data = self._get_block(self.curr_block)
            (
                cell_idx_block,
                modality_idx_block,
                study_idx_block,
                sample_idx_block,
                X_block,
                local_l_mean_block,
                local_l_var_block
            ) = self._block_data
            
            if hasattr(X_block, 'toarray'):
                X_curr = np.squeeze(np.asarray(X_block[self.cell_counter, :].toarray()))
            else:
                X_curr = np.squeeze(np.asarray(X_block[self.cell_counter, :]))
            if self.pretrained_gene_indices is not None:
                X_curr_full = np.zeros(len(self.gene_list), dtype=np.int32)
                X_curr_full[self.pretrained_gene_indices] = X_curr
                X_curr = np.squeeze(X_curr_full)

            sample_idx = sample_idx_block[self.cell_counter]
            modality_idx = modality_idx_block[self.cell_counter]
            study_idx = study_idx_block[self.cell_counter]
            cell_idx = cell_idx_block[self.cell_counter]
            local_l_mean = local_l_mean_block[self.cell_counter]
            local_l_var = local_l_var_block[self.cell_counter]
            feature_presence_mask = self.feature_presence_matrix[study_idx, :]

            datum = self._make_datum(
                X_curr, cell_idx, feature_presence_mask, modality_idx, study_idx, sample_idx, local_l_mean, local_l_var
            )
            if (self.cell_counter + 1) == len(cell_idx_block):
                self.block_counter += 1
                self.cell_counter = 0
                self._block_data = None
            else:
                self.cell_counter += 1
            return datum
        else:
            raise StopIteration
            



