import copy
import os
import math
import pickle
import time
from typing import Dict, Optional, List, Union
import warnings
import zarr 
from scipy.sparse import csr_matrix
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
from ml_benchmarking.bascvi.datamodule.zarr.utils import extract_zarr_chunk


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
    Cross-compatible library calculation class for both SOMA and zarr data.
    Handles gene/cell filtering and library statistics computation.
    """
    
    def __init__(
        self,
        data_source: str,  # "soma" or "zarr"
        data_path: str,
        root_dir: str,
        genes_to_use: List[int] = None,
        calc_library: bool = False,
        batch_keys: Dict = {"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
    ):
        """Initialize the LibraryCalculator with configuration."""
        self.data_source = data_source
        self.data_path = data_path
        self.root_dir = root_dir
        self.genes_to_use = genes_to_use
        self.calc_library = calc_library
        self.batch_keys = batch_keys
        
        # Initialize attributes
        self.obs_df = None
        self.var_df = None
        self.feature_presence_matrix = None
        self.samples_list = []
        self.library_calcs = None
        self.zarr_path_hashes = [] # Initialize for zarr
        # filter_pass_ids is only needed for SOMA
        
    def load_soma_data(self):
        """Load data from SOMA experiment."""
        # Read metadata columns
        column_names = ["soma_joinid", "barcode", self.batch_keys["study"], self.batch_keys["sample"], self.batch_keys["modality"]]

        with open_soma_experiment(self.data_path) as soma_experiment:
            all_column_names = [i.name for i in soma_experiment.obs.schema]

        # Add optional columns if present
        if "nnz" in all_column_names:
            column_names.append("nnz")

        if "log_mean" in all_column_names:
            column_names.extend(["log_mean", "log_var"])

        # Ensure all columns exist
        for c in column_names:
            if c not in all_column_names:
                raise ValueError(f"Column {c} not found in soma_experiment")

        # Read obs and var dataframes
        with open_soma_experiment(self.data_path) as soma_experiment:
            self.obs_df = soma_experiment.obs.read(column_names=tuple(column_names)).concat().to_pandas()
            self.var_df = soma_experiment.ms["RNA"].var.read(column_names=("soma_joinid", "gene",)).concat().to_pandas()

            # Ensure samples are unique across studies
            temp_df = self.obs_df.drop_duplicates(subset=[self.batch_keys["study"], self.batch_keys['sample']])
            assert temp_df[self.batch_keys["sample"]].nunique() == temp_df.shape[0], "Samples are not unique across studies"

            # Warn if barcodes are not unique
            if self.obs_df["barcode"].nunique() != self.obs_df.shape[0]:
                warnings.warn("barcodes in obs are not unique, making them unique by prefixing with study name + '__'")

            # Create categorical indices for batch keys
            
            self.obs_df["modality_idx"] = self.obs_df["modality_idx"] if self.batch_keys["modality"] == "modality_idx" else self.obs_df[self.batch_keys["modality"]].astype('category').cat.codes
            self.obs_df["study_idx"] = self.obs_df["study_idx"] if self.batch_keys["study"] == "study_idx" else self.obs_df[self.batch_keys["study"]].astype('category').cat.codes
            self.obs_df["sample_idx"] = self.obs_df["sample_idx"] if self.batch_keys["sample"] == "sample_idx" else self.obs_df[self.batch_keys["sample"]].astype('category').cat.codes

            self.feature_presence_matrix = soma_experiment.ms["RNA"]["feature_presence_matrix"].read().coos(
                shape=(self.obs_df.sample_idx.nunique(), soma_experiment.ms["RNA"].var.count)).concat().to_scipy().toarray()

        self.samples_list = sorted(self.obs_df["sample_idx"].unique().tolist())
        
    def load_zarr_data(self, zarr_dirs):

        """Load data from zarr files using zarr API only (no anndata.read_zarr)."""
        # Load var from first file
        z0 = zarr.open(zarr_dirs[0], mode='r')
        var_group = z0["var"]
        var_dict = {}
        for col_name in var_group.array_keys():
            var_dict[col_name] = var_group[col_name][...]
        var_df = pd.DataFrame(var_dict)
        self.var_df = var_df.reset_index(drop=True)
        obs_dfs = []
        feature_presence_rows = []
        zarr_path_hashes = []
        # Use the reference gene list as self.genes_to_use (should be a list of gene names)
        reference_gene_list = self.genes_to_use
        for i, zarr_path in enumerate(zarr_dirs):
            study_name = zarr_path.split("/")[-1].split(".")[0]
            z = zarr.open(zarr_path, mode='r')
            obs_group = z['obs']
            
            obs_dict = {}
            for col_name in obs_group.array_keys():
                obs_dict[col_name] = obs_group[col_name][...]

            obs_dict['__row_idx'] = np.arange(obs_dict['barcode'].shape[0])
            obs = pd.DataFrame(obs_dict)

            n_obs = obs.shape[0]
            zarr_path_hash = hashlib.md5(zarr_path.encode()).hexdigest()
            zarr_path_hashes.append(zarr_path_hash)

            print(study_name,obs.shape[0])

            file_obs = pd.DataFrame({
                'soma_joinid': range(i * 10000, i * 10000 + n_obs),
                'barcode': obs['barcode'],
                'study_name': study_name,
                '__row_idx': obs['__row_idx'],
                'scrnaseq_protocol': 'zarr_protocol',
                'modality_idx': 0,
                'study_idx': i,
                'sample_idx': obs['sample_idx'],
                'nnz': obs['nnz'],
                'zarr_path': zarr_path,
                'zarr_path_hash': zarr_path_hash
            })
            
            obs_dfs.append(file_obs)
            
            # Feature presence: use var['gene'] in this file
            file_genes = set([str(g).lower() for g in z['var']['gene'][...]] )
            present = np.array([g in file_genes for g in reference_gene_list], dtype=bool)
            feature_presence_rows.append(present)

        self.obs_df = pd.concat(obs_dfs, ignore_index=True)
        self.feature_presence_matrix = np.stack(feature_presence_rows, axis=0)
        self.samples_list = list(range(len(zarr_dirs)))
        self.zarr_path_hashes = zarr_path_hashes

    def filter_and_generate_library_calcs(self, iterative=True):
        """
        Filter cells and compute per-sample library statistics (mean/var of log counts).
        For zarr, process one file/sample at a time and X in row chunks.
        For soma, process all samples, with caching and resuming support.
        """
        filter_pass_ids_path = os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_ids.pkl')
        l_means_vars_path = os.path.join(self.root_dir, "cached_calcs_and_filter", 'l_means_vars.csv')

        def compute_log_stats(X, chunk_size=1000):
            n_rows = X.attrs["shape"][0]
            log_means = []
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                X_chunk = extract_zarr_chunk(X, start, stop)
                gene_counts = np.array((X_chunk > 0).sum(axis=1)).ravel()
                cell_mask = gene_counts > 300
                if np.any(cell_mask):
                    log_means.append(log_mean(X_chunk[cell_mask, :]))
            mean_val = float(np.mean(log_means)) if log_means else 0.0
            var_val = float(np.var(log_means)) if log_means else 1.0
            return mean_val, var_val

        print("Generating sample metadata...")

        if self.data_source == "zarr":
            zarr_dirs = [str(p) for p in Path(self.data_path).iterdir() if p.is_dir() and p.name.endswith('.zarr')]
            rows = []
            for sample_idx, zarr_path in enumerate(zarr_dirs):
                z = zarr.open(zarr_path, mode='r')
                X = z['X']
                mean_val, var_val = compute_log_stats(X)
                zarr_path_hash = hashlib.md5(zarr_path.encode()).hexdigest()
                study_name = zarr_path.split("/")[-1].split(".")[0]
                n_rows = X.attrs["shape"][0]
                for row_idx in range(n_rows):
                    rows.append({
                        "sample_idx": sample_idx,
                        "library_log_means": mean_val,
                        "library_log_vars": var_val,
                        "zarr_path_hash": zarr_path_hash,
                        "zarr_path": zarr_path,
                        "study_name": study_name,
                        "__row_idx": row_idx
                    })
            self.library_calcs = pd.DataFrame(rows)
            self.library_calcs.to_csv(l_means_vars_path, index=False)
        else:
            # --- SOMA logic: load or compute, always use local variables ---
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

        # Genes present in both sample and genes_to_use
        sample_gene_ids = np.asarray(list(set(np.where(feature_presence_matrix)[0]).intersection(set(self.genes_to_use))))

        # Get soma_joinids for this sample
        soma_ids_in_sample = self.obs_df[self.obs_df["sample_idx"] == sample_idx]["soma_joinid"].values.tolist()

        # Skip if no cells
        if len(soma_ids_in_sample) < 1:
            print(f"skipping calcs for {sample_idx}, not enough cells")
            self.l_means.append(0)
            self.l_vars.append(1)
            samples_run.append(sample_idx)
            return

        # Try to read counts for this sample
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

        # Filter cells with enough genes
        gene_counts = (X_curr != 0).sum(axis=1).flatten()
        cell_mask = gene_counts > 300
        X_curr = X_curr[cell_mask, :]

        print(f"sample {sample_idx}, X shape: {X_curr.shape}")

        # Compute stats
        self.l_means.append(log_mean(X_curr))
        self.l_vars.append(log_var(X_curr))
        samples_run.append(sample_idx)

        # Update filter pass list
        filter_pass_ids.extend(np.array(soma_ids_in_sample)[cell_mask].tolist())
        
    def _process_zarr_sample(self, sample_idx, filter_pass_ids, samples_run):
        """Process a single zarr sample using zarr directly (no anndata)."""

        zarr_dirs = [str(p) for p in Path(self.data_path).iterdir() if p.is_dir() and p.name.endswith('.zarr')]
        zarr_path = zarr_dirs[sample_idx]

        z = zarr.open(zarr_path, mode='r')
        X = z['X']
        n_rows = X.shape[0]
        chunk_size = 1000
        cell_mask_all = []
        log_means = []

        for start in range(0, n_rows, chunk_size):

            stop = min(start + chunk_size, n_rows)
            X_chunk = X[start:stop, :]
            gene_counts = np.sum(X_chunk > 0, axis=1)
            cell_mask = gene_counts > 300
            cell_mask_all.extend(cell_mask.tolist())
            if np.any(cell_mask):

                log_means.append(log_mean(X_chunk[cell_mask, :]))

        mean_val = float(np.mean(log_means)) if log_means else 0.0
        var_val = float(np.var(log_means)) if log_means else 1.0
        
        print(f"sample {sample_idx}, # passing cells: {np.sum(cell_mask_all)}")
        
        self.l_means.append(mean_val)
        self.l_vars.append(var_val)
        
        samples_run.append(sample_idx)
        cell_ids = np.where(cell_mask_all)[0] + sample_idx * 10000
        filter_pass_ids.extend(cell_ids.tolist())

            
    def _process_soma_all_samples(self, filter_pass_ids, samples_run):
        """Process all SOMA samples at once."""
        
        with open_soma_experiment(self.data_path) as soma_experiment:
            X = soma_experiment.ms["RNA"]["X"]["row_raw"].read(coords=(self.obs_df.soma_joinid.values.tolist(), )).coos(
                shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[self.obs_df.soma_joinid.values.tolist(), :]
            X = X[:, self.genes_to_use]

            gene_counts = X.getnnz(axis=1)
            cell_mask = gene_counts > 300
            X = X[cell_mask, :]
            filtered_obs = self.obs_df.loc[cell_mask]
            filter_pass_ids.extend(self.obs_df["soma_joinid"].to_numpy()[cell_mask].tolist())

            for sample_idx in tqdm(self.samples_list):
                rows = (filtered_obs["sample_idx"] == sample_idx).values
                cols = self.feature_presence_matrix[sample_idx, :].astype(bool)
                sub_X = X[rows, :][:, cols]
                self.l_means.append(log_mean(sub_X))
                self.l_vars.append(log_var(sub_X))
                samples_run.append(sample_idx)

        self.library_calcs = pd.DataFrame({"sample_idx": samples_run,
                                           "library_log_means": self.l_means,
                                           "library_log_vars": self.l_vars})
        
    def _process_zarr_all_samples(self, filter_pass_ids, samples_run):
        """Process all zarr samples at once (now just calls iterative logic)."""
        # This is now just a wrapper for the iterative logic above
        self.filter_and_generate_library_calcs(iterative=True)
        
    def setup(self, zarr_dirs=None):
        """Setup the library calculator by loading data and computing library statistics."""
        if self.data_source == "soma":
            self.load_soma_data()
        else:  # zarr
            self.load_zarr_data(zarr_dirs)
            
        if self.calc_library:
            self.filter_and_generate_library_calcs(iterative=(len(self.obs_df) > 500000))
        else:
            # Try to load existing library calculations
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