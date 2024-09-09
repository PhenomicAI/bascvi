from typing import Optional, Tuple, Union
import logging
import anndata
import pandas as pd

import numpy as np
import scipy.sparse as sp_sparse
from scipy import sparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_anndata(
    adata: anndata.AnnData,
    batch_keys: list,
    prod=False
) -> anndata.AnnData:

    """Sets up :class:`~anndata.AnnData` object for `scvi` models.

    This method will also compute the log mean and log variance per batch for the library size prior.


    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing raw counts. Rows represent cells, columns represent features (genes).
    batch_key : Optional[str], optional
        key in `adata.obs` for batch information. Categories will automatically be converted into integer

        If `None`, assigns the same batch to all the data. By default None

    Returns
    -------
    anndata.AnnData
        Adds the following fields to adata:
        .uns['batch_id_to_label']
            batch_id_to_label dictionary for batch
        .obs['local_l_mean']
            per batch library size mean
        .obs['local_l_var']
            per batch library size variance
        .obs['batch']
            batch encoded as integers
    """
    
    if not batch_keys:
        logger.info("No batch_key inputted, assuming all cells are same batch")
        batch_keys = ["batch_1"]
        adata.obs["batch_1"] = np.zeros(adata.shape[0], dtype=np.int64)
        
        log_counts = np.log(adata.X.sum(axis=1))        
        adata.obs["l_mean_batch_1"] = np.mean(log_counts).astype(np.float32)
        adata.obs["l_var_batch_1"] = np.var(log_counts).astype(np.float32)

    if prod:
        logger.info('Using batches from adata.obs["{}"]'.format("|".join(batch_keys)))

        # adds local_l_mean_key and local_l_var_key to adata.obs
        adata.obs['int_index'] = list(range(adata.shape[0]))
        log_counts = np.log(adata.X.sum(axis=1))

        for i in range(len(batch_keys)):
            print('batch-key:',i)
            header_m = "l_mean_batch_" + str(i+1)
            adata.obs[header_m] = np.mean(log_counts).astype(np.float32)
            header_v = "l_var_batch_" + str(i+1)
            adata.obs[header_v] = np.var(log_counts).astype(np.float32)
    else:
        logger.info('Using batches from adata.obs["{}"]'.format("|".join(batch_keys)))

        # adds local_l_mean_key and local_l_var_key to adata.obs
        adata.obs['int_index'] = list(range(adata.shape[0]))
    
        for i in range(len(batch_keys)):

            header_m = "l_mean_batch_" + str(i+1)
            adata.obs[header_m] = adata.obs.groupby(batch_keys[i])["int_index"].transform(log_mean, adata.X)
            header_v = "l_var_batch_" + str(i+1)
            adata.obs[header_v] = adata.obs.groupby(batch_keys[i])["int_index"].transform(log_var, adata.X)
        
    return adata

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
