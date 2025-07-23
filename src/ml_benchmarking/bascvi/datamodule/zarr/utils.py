import numpy as np
from scipy.sparse import csr_matrix
import zarr
import os

def extract_zarr_chunk(X, start, stop):
    """Extract a block of rows from a zarr CSR matrix in a memory-efficient way."""
    data = X["data"]
    indices = X["indices"]
    indptr = X["indptr"]
    shape = tuple(X.attrs["shape"])  # (n_obs, n_vars)

    indptr_start = indptr[start]
    indptr_stop = indptr[stop]

    data_slice = data[indptr_start:indptr_stop]
    indices_slice = indices[indptr_start:indptr_stop]
    indptr_slice = indptr[start:stop + 1] - indptr_start

    X_chunk = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(stop - start, shape[1]))
    return X_chunk


def extract_zarr_row(X, row_idx):
    """Extract a single row from a zarr CSR matrix as a dense numpy array."""
    data = X["data"]
    indices = X["indices"]
    indptr = X["indptr"]
    shape = tuple(X.attrs["shape"])
    start = indptr[row_idx]
    stop = indptr[row_idx + 1]
    row_data = data[start:stop]
    row_indices = indices[start:stop]
    row = np.zeros(shape[1], dtype=row_data.dtype)
    row[row_indices] = row_data
    return row


def get_or_create_feature_presence_matrix(zarr_dirs, reference_gene_list, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    matrix = []
    ref_gene_to_idx = {g: i for i, g in enumerate(reference_gene_list)}
    for zarr_path in zarr_dirs:
        z = zarr.open(zarr_path, mode='r')
        zarr_genes = [str(g).lower() for g in z['var']['gene'][...]]
        presence = np.zeros(len(reference_gene_list), dtype=bool)
        for g in zarr_genes:
            if g in ref_gene_to_idx:
                presence[ref_gene_to_idx[g]] = True
        matrix.append(presence)
    matrix = np.stack(matrix, axis=0)
    np.save(cache_path, matrix)
    return matrix 