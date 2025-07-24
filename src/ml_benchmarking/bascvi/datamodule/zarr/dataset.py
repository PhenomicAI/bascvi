import math
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset
import anndata as ad
import random
import os

class ZarrDataset(IterableDataset):
    """Dataset for block-based loading from shuffled AnnData Zarr blocks."""
    def __init__(
        self,
        data_root_dir: str,
        gene_list: Optional[List[str]] = None,
        num_workers: int = 1,
        batch_size: int = 64,
        predict_mode: bool = False,
    ):
        assert os.path.isdir(data_root_dir), f"data_root_dir {data_root_dir} does not exist or is not a directory."
        block_files = sorted([
            os.path.join(data_root_dir, f)
            for f in os.listdir(data_root_dir)
            if f.endswith('.zarr')
        ])
        assert len(block_files) > 0, f"No .zarr files found in {data_root_dir}."
        self.block_files = block_files
        self.gene_list = gene_list
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.predict_mode = predict_mode
        self._len = None  # Optionally, sum of all block lengths

    def __len__(self):
        if self._len is not None:
            return self._len
        # Optionally, compute total length by summing all block sizes
        total = 0
        for f in self.block_files:
            adata = ad.read_zarr(f)
            total += adata.n_obs
        self._len = total
        return total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        # Assign blocks to this worker
        blocks_per_worker = [self.block_files[i] for i in range(len(self.block_files)) if i % num_workers == worker_id]
        # Optionally shuffle block order per epoch
        random.shuffle(blocks_per_worker)
        for block_path in blocks_per_worker:
            adata = ad.read_zarr(block_path)
            n_obs = adata.n_obs
            # Optionally, shuffle rows within block
            indices = np.arange(n_obs)
            np.random.shuffle(indices)
            for i in range(0, n_obs, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X = adata.X[batch_idx, :]
                obs = adata.obs.iloc[batch_idx]
                # Convert to torch tensor
                X_tensor = torch.tensor(X.A if hasattr(X, 'A') else X, dtype=torch.float32)
                # Optionally, add more fields from obs
                batch = {
                    'x': X_tensor,
                    'obs': obs.reset_index(drop=True),
                }
                yield batch


