import math
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import zarr
from ml_benchmarking.bascvi.datamodule.zarr.utils import extract_zarr_row

class ZarrDataset(IterableDataset):
    """Custom torch dataset to get data from zarr in tensor form for pytorch modules."""
    
    def __init__(
        self,
        file_paths,
        reference_gene_list,
        zarr_len_dict,
        num_batches,
        num_workers,
        block_size=1000,
        predict_mode=False,
        feature_presence_matrix=None,
        library_calcs=None,
        num_modalities=None,
        num_studies=None,
        num_samples=None,
    ):
        self.reference_gene_list = reference_gene_list
        self.num_files = len(file_paths)
        self.file_paths = file_paths
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.block_size = block_size
        self.predict_mode = predict_mode
        self.zarr_len_dict = zarr_len_dict
        self._len = sum(zarr_len_dict[p] for p in file_paths)
        self.file_counter = 0
        self.row_counter = 0
        self.current_file = None
        self.current_z = None
        self.current_var = None
        self.current_obs = None
        self.current_X = None
        self.current_gene_indices = None
        self.rows_in_file = 0
        self.is_sparse = False
        self.feature_presence_matrix = feature_presence_matrix
        self.library_calcs = library_calcs
        self.num_modalities = num_modalities
        self.num_studies = num_studies
        self.num_samples = num_samples

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
        self.file_counter = self.start_file
        self.row_counter = 0
        self.current_file = None
        self.current_z = None
        self.current_var = None
        self.current_obs = None
        self.current_X = None
        self.current_gene_indices = None
        self.rows_in_file = 0
        self.is_sparse = False
        return self

    def _calc_start_end(self, worker_id):
        if self.num_files <= self.num_workers:
            start_file = worker_id
            end_file = worker_id + 1
        else:
            num_files_per_worker = math.floor(self.num_files / self.num_workers)
            start_file = worker_id * num_files_per_worker
            end_file = start_file + num_files_per_worker
            if worker_id + 1 == self.num_workers:
                end_file = self.num_files
        return (start_file, end_file)

    def _load_file(self, file_idx):
        z = zarr.open(self.file_paths[file_idx], mode='r')
        var = z['var']
        obs = z['obs']
        X = z['X']
        var_genes = [str(g).lower() for g in var['gene'][...]]
        gene_indices = [var_genes.index(g) for g in self.reference_gene_list if g in var_genes]
        self.current_z = z
        self.current_var = var
        self.current_obs = obs
        self.current_X = X
        self.current_gene_indices = gene_indices
        self.rows_in_file = X.shape[0] if hasattr(X, 'shape') else X.attrs['shape'][0]
        self.row_counter = 0
        self.is_sparse = all(k in X for k in ['data', 'indices', 'indptr'])

    def __next__(self):
        while self.file_counter < self.end_file:
            if self.current_file != self.file_paths[self.file_counter]:
                self._load_file(self.file_counter)
                self.current_file = self.file_paths[self.file_counter]
            if self.row_counter < self.rows_in_file:
                i = self.row_counter
                # Always create a full-length vector for the reference gene list
                X_full = np.zeros(len(self.reference_gene_list), dtype="int32")
                if self.is_sparse:
                    row = extract_zarr_row(self.current_X, i)
                    # Fill only the genes present in the zarr file
                    for j, gidx in enumerate(self.current_gene_indices):
                        X_full[j] = row[gidx]
                else:
                    row = np.array(self.current_X[i, :])
                    for j, gidx in enumerate(self.current_gene_indices):
                        X_full[j] = row[gidx]
                X_curr = X_full
                # Extract metadata from obs
                obs = self.current_obs
                soma_joinid = int(obs['soma_joinid'][i]) if 'soma_joinid' in obs else i
                cell_idx = i
                sample_idx = int(obs['sample_idx'][i]) if 'sample_idx' in obs else 0
                modality_idx = int(obs['modality_idx'][i]) if 'modality_idx' in obs else 0
                study_idx = int(obs['study_idx'][i]) if 'study_idx' in obs else 0
                if self.feature_presence_matrix is not None:
                    feature_presence_mask = self.feature_presence_matrix[sample_idx, :]
                else:
                    feature_presence_mask = np.ones(len(self.reference_gene_list), dtype=bool)
                base = {
                    "x": torch.from_numpy(X_curr),
                    "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
                    "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
                    "feature_presence_mask": torch.from_numpy(feature_presence_mask),
                }
                if self.predict_mode:
                    self.row_counter += 1
                    return base
                one_hot_modality = F.one_hot(torch.tensor(modality_idx, dtype=torch.long), num_classes=self.num_modalities).float() if self.num_modalities else torch.tensor([1.0])
                one_hot_study = F.one_hot(torch.tensor(study_idx, dtype=torch.long), num_classes=self.num_studies).float() if self.num_studies else torch.tensor([1.0])
                one_hot_sample = F.one_hot(torch.tensor(sample_idx, dtype=torch.long), num_classes=self.num_samples).float() if self.num_samples else torch.tensor([1.0])
                if self.library_calcs is not None and sample_idx in self.library_calcs.index:
                    local_l_mean = self.library_calcs.loc[sample_idx, "library_log_means"]
                    local_l_var = self.library_calcs.loc[sample_idx, "library_log_vars"]
                else:
                    local_l_mean = 0.0
                    local_l_var = 1.0
                base.update({
                    "modality_vec": one_hot_modality,
                    "study_vec": one_hot_study,
                    "sample_vec": one_hot_sample,
                    "local_l_mean": torch.tensor(local_l_mean),
                    "local_l_var": torch.tensor(local_l_var),
                })
                self.row_counter += 1
                return base
            else:
                self.file_counter += 1
                self.current_file = None
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

