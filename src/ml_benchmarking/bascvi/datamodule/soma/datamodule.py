import copy
import os
import math
import pickle
import time
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import tiledbsoma as soma
from pyarrow.lib import ArrowInvalid
import pytorch_lightning as pl
from torch.utils.data import DataLoader, get_worker_info

from ml_benchmarking.bascvi.datamodule.soma.dataset import TileDBSomaTorchIterDataset
from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.datamodule.library_calculations import LibraryCalculator, log_mean, log_var, staggered_worker_init


class TileDBSomaIterDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preprocessing single-cell data from TileDB-SOMA.
    Handles gene/cell filtering, batching, and DataLoader creation for training, validation, and prediction.
    """

    def __init__(
        self,
        soma_experiment_uri,
        root_dir,
        max_cells_per_sample=None,
        pretrained_gene_list=None,
        genes_to_use_path=None,
        genes_to_use_hvg=None,
        cells_to_use_path=None,
        barcodes_to_use_path=None,
        calc_library=False,
        block_size=1000,
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        verbose: bool = False,
        exclude_ribo_mito=True,
        train_column: str = None,
        random_seed: int = 42,
        batch_keys={"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
    ):
        """Initialize the DataModule with configuration and file paths."""
        super().__init__()

        self.soma_experiment_uri = soma_experiment_uri
        self.dataloader_args = dataloader_args
        self.root_dir = root_dir
        self.max_cells_per_sample = max_cells_per_sample
        self.pretrained_gene_list = pretrained_gene_list
        self.genes_to_use_path = genes_to_use_path
        self.genes_to_use_hvg = genes_to_use_hvg
        self.cells_to_use_path = cells_to_use_path
        self.barcodes_to_use_path = barcodes_to_use_path
        self.calc_library = calc_library
        self.block_size = block_size
        self.pretrained_batch_size = pretrained_batch_size
        self.verbose = verbose
        self.exclude_ribo_mito = exclude_ribo_mito
        self.train_column = train_column
        self.random_seed = random_seed
        self.batch_keys = batch_keys


    def filter_and_generate_library_calcs(self, iterative=True):
        """
        Filter cells and compute per-sample library statistics (mean/var of log counts).
        Uses the cross-compatible LibraryCalculator.
        """
        # Initialize library calculator
        library_calc = LibraryCalculator(
            data_source="soma",
            data_path=self.soma_experiment_uri,
            root_dir=self.root_dir,
            genes_to_use=self.genes_to_use,
            calc_library=self.calc_library,
            batch_keys=self.batch_keys
        )
        
        # Setup and run library calculations
        library_calc.setup()
        
        # Get results
        self.library_calcs = library_calc.get_library_calcs()
        self.filter_pass_soma_ids = library_calc.get_filter_pass_ids()
        
        # Update obs_df and feature_presence_matrix from library calculator
        self.obs_df = library_calc.obs_df
        self.var_df = library_calc.var_df
        self.feature_presence_matrix = library_calc.feature_presence_matrix
        self.samples_list = library_calc.samples_list


    def setup(self, stage: Optional[str] = None):
        """
        Prepare the dataset for training, validation, or prediction.
        Handles gene/cell filtering, batching, and dataset creation.
        """
        # Read metadata columns
        column_names = ["soma_joinid", "barcode", self.batch_keys["study"], self.batch_keys["sample"], self.batch_keys["modality"]]

        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            all_column_names = [i.name for i in soma_experiment.obs.schema]

        # Add optional columns if present
        if "nnz" in all_column_names:
            column_names.append("nnz")

        if "log_mean" in all_column_names:
            column_names.extend(["log_mean", "log_var"])

        if self.train_column:
            column_names.append(self.train_column)

        # Ensure all columns exist
        for c in column_names:
            if c not in all_column_names:
                raise ValueError(f"Column {c} not found in soma_experiment")

        # Read obs and var dataframes
        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
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

            self.num_modalities = int(self.obs_df["modality_idx"].max() + 1)
            self.num_studies = int(self.obs_df["study_idx"].max() + 1)
            self.num_samples = int(self.obs_df["sample_idx"].max() + 1)

            self.feature_presence_matrix = soma_experiment.ms["RNA"]["feature_presence_matrix"].read().coos(
                shape=(self.obs_df.sample_idx.nunique(), soma_experiment.ms["RNA"].var.count)).concat().to_scipy().toarray()

        # --- Genes to use ---
        if self.genes_to_use_path:
            # For macrogenes, ensure gene names are lowercase
            self.var_df["gene"] = self.var_df["gene"].str.lower()

            with open(self.genes_to_use_path, "r") as f:
                self.genes_to_use = f.read().split("\n")
                self.genes_to_use = [g.lower() for g in self.genes_to_use]
                self.genes_to_use = list(set(self.genes_to_use).intersection(set(self.var_df["gene"].tolist())))
            self.var_df = self.var_df.set_index("gene")
            self.genes_to_use = list(set(self.var_df.loc[self.genes_to_use, :]["soma_joinid"].values.tolist()))
            self.var_df = self.var_df.reset_index()

            print("read gene list with length ", len(self.genes_to_use), len(set(self.genes_to_use)))

        elif self.genes_to_use_hvg:
            # Use highly variable genes
            print("Running HVG with n_hvg = ", self.genes_to_use_hvg)
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                with soma_experiment.axis_query(
                    measurement_name="RNA", obs_query=soma.AxisQuery(coords=(None, None))
                ) as query:
                    adata: sc.AnnData = query.to_anndata(
                        X_name="row_raw",
                        column_names={"obs": ["soma_joinid"], "var": ["soma_joinid", "gene"]},
                    )
                    sc.pp.filter_cells(adata, min_genes=300)
                    sc.pp.filter_genes(adata, min_cells=3)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(adata, n_top_genes=self.genes_to_use_hvg)
                    self.genes_to_use = adata.var[adata.var["highly_variable"]]["soma_joinid"].values.tolist()
        else:
            # Use all genes
            print("Using all genes")
            self.genes_to_use = self.var_df["soma_joinid"].values.tolist()

        # Exclude mitochondrial/ribosomal genes if requested
        if self.exclude_ribo_mito:
            exclusion_regex = r'^(MT-|RPL|RPS|MRPL|MRPS)'
            genes_to_exclude = set(self.var_df[self.var_df["gene"].str.contains(exclusion_regex)]["soma_joinid"].values)
            self.genes_to_use = list(set(self.genes_to_use).difference(genes_to_exclude))

        # Handle pretrained gene list if provided
        if self.pretrained_gene_list:
            
            self.pretrained_gene_list = [g.lower() for g in self.pretrained_gene_list]
            self.pretrained_gene_ids = self.var_df[self.var_df["gene"].str.lower().isin(self.pretrained_gene_list)].soma_joinid.values.tolist()
            self.genes_to_use = list(set(self.pretrained_gene_ids).intersection(set(self.genes_to_use)))
            print(f"Using pretrained gene list found {100 * len(self.genes_to_use) / len(self.pretrained_gene_list)}% overlap")
            self.var_df_sub = self.var_df[self.var_df["soma_joinid"].isin(self.genes_to_use)].sort_values("gene")
            self.soma_gene_name_list = self.var_df_sub["gene"].str.lower().values.tolist()
            self.genes_to_use = self.var_df_sub["soma_joinid"].values.tolist()
            self.pretrained_gene_indices = [self.pretrained_gene_list.index(gene) for gene in self.soma_gene_name_list]
            self.num_genes = len(self.pretrained_gene_list)
            print("pretrained gene indices: ", len(self.pretrained_gene_indices))
            print("pretrained gene list: ", len(self.pretrained_gene_list))
            print("max gene index: ", max(self.pretrained_gene_indices))

        else:
            self.var_df_sub = self.var_df[self.var_df["soma_joinid"].isin(self.genes_to_use)].sort_values("gene")
            self.soma_gene_name_list = self.var_df_sub["gene"].values.tolist()
            self.genes_to_use = self.var_df_sub["soma_joinid"].values.tolist()
            self.gene_list = self.soma_gene_name_list
            self.pretrained_gene_indices = None
            self.num_genes = len(self.genes_to_use)

        # --- Cells to use ---
        self.cells_to_use = None

        if self.barcodes_to_use_path:
            if self.cells_to_use is not None:
                raise ValueError("cannot use both cells_to_use and barcodes_to_use")
            with open(self.barcodes_to_use_path, "rb") as f:
                barcodes = pickle.load(f)
            if len(set(barcodes)) != len(barcodes):
                raise ValueError("barcodes in given barcode list are not unique")
            self.cells_to_use = self.obs_df.loc[self.obs_df["barcode"].isin(barcodes)]["soma_joinid"].values.tolist()
            print("read barcode list with", len(self.cells_to_use), "cells found in obs.")

        if self.train_column:
            if self.cells_to_use is not None:
                raise ValueError("cannot use both train_column and barcodes_to_use/cell_to_use")
            self.cells_to_use = self.obs_df.loc[self.obs_df[self.train_column]]["soma_joinid"].values.tolist()
            print("read cell list with length ", len(self.cells_to_use))

        if self.cells_to_use is None:
            print("Using all cells found in obs")
            self.cells_to_use = self.obs_df["soma_joinid"].values.tolist()

        self.samples_list = sorted(self.obs_df["sample_idx"].unique().tolist())
        self.num_samples = len(self.samples_list)

        # Downsample if needed
        if self.max_cells_per_sample:
            self.obs_df = self.obs_df.groupby('sample_idx').apply(
                lambda x: x.sample(n=self.max_cells_per_sample, random_state=self.random_seed) if x.shape[0] > self.max_cells_per_sample else x
            ).reset_index(drop=True)
            print("\tdownsampled to ", self.obs_df.shape[0], " cells")

        # Filter cells
        self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.cells_to_use)]
        print("Before filtering: ", self.obs_df.shape[0], " cells")

        # Filter by nnz if present
        if "nnz" in self.obs_df.columns:
            pass_filter = self.obs_df["nnz"] > 300
            print("Filtering cells with nnz < 300, ", pass_filter.sum(), " cells")
            self.obs_df = self.obs_df[pass_filter]

        self.feature_presence_matrix = self.feature_presence_matrix[:, self.genes_to_use]

        # --- Library calculations ---
        try:
            if "log_mean" in self.obs_df.columns:
                self.library_calcs = self.obs_df[["sample_idx", "log_mean", "log_var"]].drop_duplicates(subset=["sample_idx"])
                self.library_calcs = self.library_calcs.set_index("sample_idx")
                self.library_calcs.columns = ["library_log_means", "library_log_vars"]
            else:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    self.library_calcs = soma_experiment.ms["RNA"]["sample_library_calcs"].read().concat().to_pandas()
                    self.library_calcs = self.library_calcs.set_index("sample_idx")
        except Exception:
            self.filter_and_generate_library_calcs(iterative=(self.obs_df.shape[0] > 500000))
            print(len(set(self.filter_pass_soma_ids)), " cells passed final filter.")
            self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.filter_pass_soma_ids)]

        # Assign cell_idx and shuffle
        self.obs_df['cell_idx'] = range(self.obs_df.shape[0])

        if stage != "predict":
            self.obs_df = self.obs_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Calculate block sizes
        MIN_VAL_BLOCKS = 2
        if self.block_size > self.obs_df.shape[0] // 5:
            self.block_size = self.obs_df.shape[0] // 5
        self.num_total_blocks = math.ceil(self.obs_df.shape[0] / self.block_size)

        # Adjust num_workers if needed
        if self.num_total_blocks < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = self.num_total_blocks

        self.num_cells = self.obs_df.shape[0]
        self.num_batches = self.num_modalities + self.num_studies + self.num_samples

        print('# Blocks: ', self.num_total_blocks)
        print('# Genes: ', self.num_genes)
        print('# Total Cells: ', self.num_cells)
        print('# Modalities: ', self.num_modalities)
        print('# Studies: ', self.num_studies)
        print('# Samples: ', self.num_samples)

        self.batch_level_sizes = [self.num_modalities, self.num_studies, self.num_samples]

        print("Obs has ", self.obs_df.shape[0], " cells, ", self.obs_df.soma_joinid.nunique(), " unique soma_joinids")
        assert self.obs_df.soma_joinid.nunique() == self.obs_df.shape[0]
        assert self.obs_df.cell_idx.nunique() == self.obs_df.shape[0]

        self.obs_df['soma_joinid'] = self.obs_df['soma_joinid'].astype('int64')
        self.obs_df['cell_idx'] = self.obs_df['cell_idx'].astype('int64')
        self.obs_df = self.obs_df[['cell_idx','soma_joinid','modality_idx','study_idx','sample_idx']]

        # --- Dataset creation ---
        if stage == "fit":
            print("Stage = Fitting")
            self.val_blocks = max(self.num_total_blocks//5, MIN_VAL_BLOCKS)
            self.train_blocks = self.num_total_blocks - self.val_blocks
            print('# Blocks: ', self.num_total_blocks, ' # for Training: ', self.train_blocks)
            self.train_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df[ : self.train_blocks * self.block_size],
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                block_size=self.block_size,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
                num_workers=self.dataloader_args['num_workers'],
                verbose=self.verbose
            )
            self.val_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df[self.train_blocks * self.block_size : ],
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                block_size=self.block_size,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
                num_workers=self.dataloader_args['num_workers'],
                verbose=self.verbose
            )
        elif stage == "predict":
            print("Stage = Predicting")
            self.pred_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df,
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                block_size=self.block_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=True,
                pretrained_gene_indices=self.pretrained_gene_indices,
                verbose=self.verbose
            )


    def train_dataloader(self):
        """Return DataLoader for training dataset."""
        return DataLoader(self.train_dataset, persistent_workers=True, worker_init_fn=staggered_worker_init, **self.dataloader_args)

    def val_dataloader(self):
        """Return DataLoader for validation dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, persistent_workers=True, worker_init_fn=staggered_worker_init, **loader_args)

    def predict_dataloader(self):
        """Return DataLoader for prediction dataset."""
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)


    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move batch to the specified device."""
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch


