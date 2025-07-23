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
        self.zarr_len_dict = zarr_len_dict
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.block_size = block_size
        self.predict_mode = predict_mode
        self.library_calcs = library_calcs
        self.num_modalities = num_modalities
        self.num_studies = num_studies
        self.num_samples = num_samples
        self._len = sum(zarr_len_dict[p] for p in file_paths)
        self.indices = None
        self.current_z = None
        self.current_var = None
        self.current_obs = None
        self.current_X = None
        self.current_gene_indices = None
        self.is_sparse = False
        import hashlib
        self.hash_to_path = {hashlib.md5(p.encode()).hexdigest(): p for p in file_paths}
        self.hash_to_idx = {h: i for i, h in enumerate(self.hash_to_path.keys())}
        # Convert feature_presence_matrix to a hash-based dict for robust lookup
        self.feature_presence_matrix = None
        if feature_presence_matrix is not None:
            # Assume order matches file_paths
            self.feature_presence_matrix = {
                h: feature_presence_matrix[i, :]
                for i, h in enumerate(self.hash_to_path.keys())
            }

    def __len__(self):
        return self._len

    def __iter__(self):
        if torch.utils.data.get_worker_info():
            worker_info = torch.utils.data.get_worker_info()
            per_worker = int(np.ceil(self._len / self.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self._len)
            self.indices = np.arange(start, end)
        else:
            self.indices = np.arange(self._len)
        self.row_counter = 0
        self.current_z = None
        self.current_var = None
        self.current_obs = None
        self.current_X = None
        self.current_gene_indices = None
        self.is_sparse = False
        self.current_file_hash = None
        return self

    def __next__(self):
        if self.row_counter >= len(self.indices):
            raise StopIteration
        idx = self.indices[self.row_counter]
        obs_row = self.library_calcs.iloc[idx]
        file_hash = obs_row['zarr_path_hash']
        row_idx = int(obs_row['__row_idx'])
        # Load zarr file if needed
        if self.current_z is None or self.current_file_hash != file_hash:
            zarr_path = self.hash_to_path[file_hash]
            z = zarr.open(zarr_path, mode='r')
            var = z['var']
            obs = z['obs']
            X = z['X']
            var_genes = [str(g).lower() for g in var['gene'][...]]
            gene_indices = [var_genes.index(g) for g in self.reference_gene_list if g in var_genes]
            feature_presence_mask = self.feature_presence_matrix[file_hash] if self.feature_presence_matrix is not None else np.ones(len(self.reference_gene_list), dtype=bool)
            self.current_z = z
            self.current_var = var
            self.current_obs = obs
            self.current_X = X
            self.current_gene_indices = gene_indices
            self.is_sparse = all(k in X for k in ['data', 'indices', 'indptr'])
            self.current_file_hash = file_hash
        else:
            feature_presence_mask = self.feature_presence_matrix[file_hash] if self.feature_presence_matrix is not None else np.ones(len(self.reference_gene_list), dtype=bool)
        X_full = np.zeros(len(self.reference_gene_list), dtype="int32")
        if self.is_sparse:
            row = extract_zarr_row(self.current_X, row_idx)
            for j, gidx in enumerate(self.current_gene_indices):
                X_full[j] = row[gidx]
        else:
            row = np.array(self.current_X[row_idx, :])
            for j, gidx in enumerate(self.current_gene_indices):
                X_full[j] = row[gidx]
        X_curr = X_full
        soma_joinid = int(obs_row['soma_joinid']) if 'soma_joinid' in obs_row else idx
        cell_idx = idx
        sample_idx = int(obs_row['sample_idx']) if 'sample_idx' in obs_row else 0
        modality_idx = int(obs_row['modality_idx']) if 'modality_idx' in obs_row else 0
        study_idx = int(obs_row['study_idx']) if 'study_idx' in obs_row else 0
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


