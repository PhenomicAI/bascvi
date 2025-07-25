import copy
import os
import math
import pickle
import time
from typing import Dict, Optional, List, Union
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import tiledbsoma as soma
from pyarrow.lib import ArrowInvalid
import anndata
import hashlib

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment


def log_mean(X):
    """Compute the mean of the log total counts per cell."""
    log_counts = np.log(X.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(X):
    """Compute the variance of the log total counts per cell."""
    log_counts = np.log(X.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var

def staggered_worker_init(worker_id):
    """Custom worker init function to stagger the initialization of DataLoader workers."""
    time.sleep(worker_id * 0)  # Currently no delay, but can be adjusted if needed

class LibraryCalculator:
    """
    Library calculation class for SOMA data only.
    Handles gene/cell filtering and library statistics computation.
    """
    
    def __init__(
        self,
        data_source: str,  # Should be "soma"
        data_path: str,
        root_dir: str,
        genes_to_use: List[int] = None,
        calc_library: bool = False,
        batch_keys: Dict = {"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
    ):
        self.data_source = data_source
        self.data_path = data_path
        self.root_dir = root_dir
        self.genes_to_use = genes_to_use
        self.calc_library = calc_library
        self.batch_keys = batch_keys
        self.obs_df = None
        self.var_df = None
        self.feature_presence_matrix = None
        self.samples_list = []
        self.library_calcs = None
        # filter_pass_ids is only needed for SOMA
    
    def load_soma_data(self):
        """Load data from SOMA experiment."""
        column_names = ["soma_joinid", "barcode", self.batch_keys["study"], self.batch_keys["sample"], self.batch_keys["modality"]]
        with open_soma_experiment(self.data_path) as soma_experiment:
            all_column_names = [i.name for i in soma_experiment.obs.schema]
        if "nnz" in all_column_names:
            column_names.append("nnz")
        if "log_mean" in all_column_names:
            column_names.extend(["log_mean", "log_var"])
        for c in column_names:
            if c not in all_column_names:
                raise ValueError(f"Column {c} not found in soma_experiment")
        with open_soma_experiment(self.data_path) as soma_experiment:
            self.obs_df = soma_experiment.obs.read(column_names=tuple(column_names)).concat().to_pandas()
            self.var_df = soma_experiment.ms["RNA"].var.read(column_names=("soma_joinid", "gene",)).concat().to_pandas()
            temp_df = self.obs_df.drop_duplicates(subset=[self.batch_keys["study"], self.batch_keys['sample']])
            assert temp_df[self.batch_keys["sample"]].nunique() == temp_df.shape[0], "Samples are not unique across studies"
            if self.obs_df["barcode"].nunique() != self.obs_df.shape[0]:
                warnings.warn("barcodes in obs are not unique, making them unique by prefixing with study name + '__'")
            self.obs_df["modality_idx"] = self.obs_df["modality_idx"] if self.batch_keys["modality"] == "modality_idx" else self.obs_df[self.batch_keys["modality"]].astype('category').cat.codes
            self.obs_df["study_idx"] = self.obs_df["study_idx"] if self.batch_keys["study"] == "study_idx" else self.obs_df[self.batch_keys["study"]].astype('category').cat.codes
            self.obs_df["sample_idx"] = self.obs_df["sample_idx"] if self.batch_keys["sample"] == "sample_idx" else self.obs_df[self.batch_keys["sample"]].astype('category').cat.codes
            self.feature_presence_matrix = soma_experiment.ms["RNA"]["feature_presence_matrix"].read().coos(
                shape=(self.obs_df.sample_idx.nunique(), soma_experiment.ms["RNA"].var.count)).concat().to_scipy().toarray()
        self.samples_list = sorted(self.obs_df["sample_idx"].unique().tolist())
        
    def filter_and_generate_library_calcs(self, iterative=True):
        """
        Filter cells and compute per-sample library statistics (mean/var of log counts).
        For soma, process all samples, with caching and resuming support.
        """
        filter_pass_ids_path = os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_ids.pkl')
        l_means_vars_path = os.path.join(self.root_dir, "cached_calcs_and_filter", 'l_means_vars.csv')
        print("Generating sample metadata...")
        if os.path.exists(filter_pass_ids_path) and os.path.exists(l_means_vars_path):
            with open(filter_pass_ids_path, 'rb') as f:
                filter_pass_ids = pickle.load(f)
            library_calcs = pd.read_csv(l_means_vars_path)
            if max(library_calcs["sample_idx"].to_list()) == max(self.samples_list):
                self.library_calcs = library_calcs.set_index("sample_idx")
                self.filter_pass_ids = set(filter_pass_ids)
                print(f"Loaded cached library calcs ({len(self.filter_pass_ids)} cells passed filter)")
                return
            else:
                l_means = library_calcs["library_log_means"].to_list()
                l_vars = library_calcs["library_log_vars"].to_list()
                samples_run = library_calcs["sample_idx"].to_list()
        else:
            filter_pass_ids, l_means, l_vars, samples_run = [], [], [], []
        for sample_idx in tqdm(self.samples_list):
            self._process_soma_sample(sample_idx, filter_pass_ids, samples_run)
            l_means.append(self.l_means[-1])
            l_vars.append(self.l_vars[-1])
        self.filter_pass_ids = set(filter_pass_ids)
        self.library_calcs = pd.DataFrame({
            "sample_idx": self.samples_list,
            "library_log_means": l_means,
            "library_log_vars": l_vars
        }).set_index("sample_idx")
        self.library_calcs.to_csv(l_means_vars_path, index=True)
        with open(filter_pass_ids_path, 'wb') as f:
            pickle.dump(self.filter_pass_ids, f)
        print(f"Computed library calcs ({len(self.filter_pass_ids)} cells passed filter)")

    def _process_soma_sample(self, sample_idx, filter_pass_ids, samples_run):
        """Process a single SOMA sample."""
        feature_presence_matrix = self.feature_presence_matrix[sample_idx, :].astype(bool)
        sample_gene_ids = np.asarray(list(set(np.where(feature_presence_matrix)[0]).intersection(set(self.genes_to_use))))
        soma_ids_in_sample = self.obs_df[self.obs_df["sample_idx"] == sample_idx]["soma_joinid"].values.tolist()
        if len(soma_ids_in_sample) < 1:
            print(f"skipping calcs for {sample_idx}, not enough cells")
            self.l_means.append(0)
            self.l_vars.append(1)
            samples_run.append(sample_idx)
            return
        try:
            with open_soma_experiment(self.data_path) as soma_experiment:
                X_curr = soma_experiment.ms["RNA"]["X"]["row_raw"].read((soma_ids_in_sample, None)).coos(
                    shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[soma_ids_in_sample, :]
                X_curr = X_curr[:, sample_gene_ids]
        except ArrowInvalid:
            print(f"skipping calcs for {sample_idx}, not enough counts")
            self.l_means.append(0)
            self.l_vars.append(1)
            samples_run.append(sample_idx)
            return
        gene_counts = (X_curr != 0).sum(axis=1).flatten()
        cell_mask = gene_counts > 300
        X_curr = X_curr[cell_mask, :]
        print(f"sample {sample_idx}, X shape: {X_curr.shape}")
        self.l_means.append(log_mean(X_curr))
        self.l_vars.append(log_var(X_curr))
        samples_run.append(sample_idx)
        filter_pass_ids.extend(np.array(soma_ids_in_sample)[cell_mask].tolist())
        
    def setup(self, zarr_dirs=None):
        """Setup the library calculator by loading data and computing library statistics."""
        self.load_soma_data()
        if self.calc_library:
            self.filter_and_generate_library_calcs()
        else:
            try:
                l_means_vars_path = os.path.join(self.root_dir, "cached_calcs_and_filter", 'l_means_vars.csv')
                if os.path.exists(l_means_vars_path):
                    self.library_calcs = pd.read_csv(l_means_vars_path)
                    self.library_calcs.set_index("sample_idx")
                    print("Loaded existing library calculations")
                else:
                    print("No existing library calculations found, creating simple ones")
                    self._create_simple_library_calcs()
            except Exception:
                print("Error loading library calculations, creating simple ones")
                self._create_simple_library_calcs()
                
    def _create_simple_library_calcs(self):
        """Create simple library calculations when detailed ones are not available."""
        self.library_calcs = pd.DataFrame({
            "sample_idx": self.samples_list,
            "library_log_means": [10.0] * len(self.samples_list),  # Default values
            "library_log_vars": [1.0] * len(self.samples_list)
        })
        self.library_calcs.set_index("sample_idx")
        
    def get_library_calcs(self):
        """Get the library calculations."""
        return self.library_calcs
        
    def get_filter_pass_ids(self):
        """Get the IDs of cells that passed filtering."""
        return self.filter_pass_ids 