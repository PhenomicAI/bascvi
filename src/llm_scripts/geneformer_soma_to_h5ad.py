import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import pickle as pkl

import re
import os

import tiledbsoma as soma
import tiledb
import scanpy as sc


from pathlib import Path

save_dir = Path("")
h5ad_name = "result.h5ad"

# load data
print("- reading soma...")



ACCESS_KEY = ""
SECRET_KEY = ""

SOMA_URI = ""

soma_experiment = soma.Experiment.open(SOMA_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

with open("/home/ubuntu/geneformer_eval/extra_pal_barcodes.pkl", "rb") as f:
    scref_train_barcodes = pkl.load(f)

obs_df = soma_experiment.obs.read().concat().to_pandas()
soma_ids = obs_df[obs_df["barcode"].isin(scref_train_barcodes)]["soma_joinid"].values.tolist()

with soma_experiment.axis_query(
    measurement_name="RNA", obs_query=soma.AxisQuery(coords=(soma_ids, None))
) as query:
    adata: sc.AnnData = query.to_anndata(
        X_name="row_raw",
        column_names={"obs": ["soma_joinid", "standard_true_celltype"], "var": ["soma_joinid", "gene"]},
    )
print("     soma read! adata shape: ", adata.shape)

# convert to ensembl
print("- converting genes...")
gene_conversion_df = pd.read_csv("./HPA_ENSEMBL_PHENOMIC.merged_versions.tsv.gz", sep="\t")
new_var = (adata.var.merge(gene_conversion_df, left_on='gene', right_on='GeneName_main')
          .reindex(columns=['soma_joinid', 'gene', 'Ensembl_main']))
new_var = new_var.rename(columns={'soma_joinid': 'soma_joinid', 'gene': 'gene', 'Ensembl_main': "ensembl_id"})
adata.var = new_var
adata.var.rename(columns={'soma_joinid': 'soma_joinid', 'gene': 'gene', 'Ensembl_main': "ensembl_id"}, inplace=True)
print("     genes converted! adata shape: ", adata.shape)

import pickle
PATH_TO_GENEFORMER_TOKEN_DICT = ""
with open(PATH_TO_GENEFORMER_TOKEN_DICT, "rb") as f:
    geneformer_token_dict = pickle.load(f)

# print(geneformer_token_dict)
print("% overlap with adata: ", len(set(geneformer_token_dict.keys()).intersection(set(adata.var["ensembl_id"].values.tolist())))/len(geneformer_token_dict.keys()))

adata.var['found_in_geneformer'] = adata.var["ensembl_id"].isin(geneformer_token_dict.keys())

# save csv of genes 
adata.var.to_csv(save_dir / "geneformer_gene_found.csv")

# cell metadata
print("- adding cell metadata...")
adata.obs["n_counts"] = adata.X.sum(axis=1)
# we already filtered
adata.obs["filter_pass"] = 1
# adata.obs["filter_pass"].astype(int)

# write h5ad
print("- writing h5ad...")
os.makedirs(save_dir, exist_ok=True)
adata.write_h5ad(save_dir / h5ad_name)
print("     h5ad written!")