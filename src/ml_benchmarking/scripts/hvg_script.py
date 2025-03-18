import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import pickle as pkl

import re
import os

from ml_benchmarking.bascvi.utils.utils import umap_calc_and_save_html
import tiledbsoma as soma

from random import sample

import scanpy as sc

import tqdm


def compute_highly_variable_genes_in_chunks(soma_experiment, chunk_size=1000, n_top_genes=4000):
    # Initialize an empty DataFrame to store the variability information of all genes
    all_variability_info = pd.DataFrame()

    n_vars = soma_experiment.ms["RNA"].var.count

    # Determine the number of chunks needed to process the genes in batches
    n_chunks = (n_vars // chunk_size) + (0 if n_vars % chunk_size == 0 else 1)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, n_vars)
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks}: genes {start} to {end}")

        # Subset the AnnData object to only include genes for the current chunk
        with soma_experiment.axis_query(
            measurement_name="RNA", obs_query=soma.AxisQuery(coords=(None, slice(start, end)))
        ) as query:
            adata_sub: sc.AnnData = query.to_anndata(
                X_name="row_raw",
                column_names={"obs": ["soma_joinid"], "var": ["soma_joinid", "gene"]},
            )

        # Compute highly variable genes for the chunk
        sc.pp.highly_variable_genes(adata_sub, batch_key='batch' if 'batch' in adata_sub.obs else None, inplace=False)

        # Extract the variability information and add it to our cumulative DataFrame
        variability_info = adata_sub.var[['highly_variable', 'means', 'dispersions', 'dispersions_norm']]
        variability_info['gene'] = adata_sub.var_names
        all_variability_info = pd.concat([all_variability_info, variability_info], ignore_index=True)

    # After processing all chunks, select the top 4000 most variable genes
    top_genes = all_variability_info.nlargest(n_top_genes, 'dispersions_norm')

    return top_genes, all_variability_info

soma_experiment = soma.Experiment.open("./data/scref/")

top_genes, all_variability_info = compute_highly_variable_genes_in_chunks(soma_experiment, chunk_size=100, n_top_genes=4000)

all_variability_info.to_csv("all_variability_info_scref.csv", index=False)

with open("top_genes_scref_all_cells.pkl", "wb") as f:
    pkl.dump(list(top_genes), f)

# obs_df = soma_experiment.obs.read(
#                     column_names=("soma_joinid",)#"standard_true_celltype", "sample_name", "study_name", "cell_type_pred", "tissue_collected"),
#                 ).concat().to_pandas()
# obs_df

# all_soma_ids = obs_df["soma_joinid"].values.tolist()
# num_hvg = 4000

# genes_to_use = None

# for i in range(50):
#     print(i)
#     sampled_soma_ids = sample(all_soma_ids, 100000)

#     with soma_experiment.axis_query(
#         measurement_name="RNA", obs_query=soma.AxisQuery(coords=(sampled_soma_ids, None))
#     ) as query:
#         adata: sc.AnnData = query.to_anndata(
#             X_name="row_raw",
#             column_names={"obs": ["soma_joinid"], "var": ["soma_joinid", "gene"]},
#         )
#         sc.pp.filter_cells(adata, min_genes=300)
#         sc.pp.filter_genes(adata, min_cells=3)
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         hvg_result = sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg, inplace=False)


#         genes_to_use.append(adata.var[adata.var["highly_variable"]]["soma_joinid"].values.tolist())

# genes_to_use = [gene for run in genes_to_use for gene in run]
