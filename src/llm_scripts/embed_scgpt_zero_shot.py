from pathlib import Path
import warnings

import scanpy as sc
import sys
sys.path.insert(0, "../")

import scgpt as scg
import matplotlib.pyplot as plt

import tiledbsoma as soma
import tiledb

import json
import pickle

import pandas as pd
from dotenv import load_dotenv

from scgpt.preprocess import Preprocessor
import torch
import os


plt.style.context('default')
warnings.simplefilter("ignore", ResourceWarning)


model_dir = Path("")
save_dir = Path("")

# make sure the save_dir exists
os.makedirs(save_dir, exist_ok=True)

# load data
import os
import tiledbsoma as soma
import tiledb

ACCESS_KEY = ""
SECRET_KEY = ""
SOMA_URI = ""

soma_experiment = soma.Experiment.open(SOMA_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

soma_ids = None


# read obs
obs_df = soma_experiment.obs.read(column_names=["soma_joinid", "barcode"]).concat().to_pandas()

# filter barcodes

with open("/home/ubuntu/scGPT/extra_pal_barcodes.pkl", "rb") as f:
    pal_barcodes = pickle.load(f)
    print(len(set(pal_barcodes)))

with open("/home/ubuntu/ml/bascvi/data/scref_train_barcodes.pkl", "rb") as f:
    scref_train_barcodes = pickle.load(f)
    print(len(set(scref_train_barcodes)))

train_barcodes = set(pal_barcodes).union(set(scref_train_barcodes))
print("Total barcodes: ", len(train_barcodes))

soma_ids = obs_df[obs_df["barcode"].isin(train_barcodes)]["soma_joinid"].values.tolist()
if len(set(soma_ids)) != len(set(train_barcodes)):
    print('ERROR: barcodes not found in soma. n=', (len(train_barcodes) - len(soma_ids)))

# get adata
with soma_experiment.axis_query(
    measurement_name="RNA", obs_query=soma.AxisQuery(coords=((soma_ids, None, None, None)))
) as query:
    adata: sc.AnnData = query.to_anndata(
        X_name="row_raw",
        column_names={"obs": ["soma_joinid", "barcode", "study_name"], "var": ["soma_joinid", "gene"]},
    )
    print(adata)

del obs_df

gene_col = "gene"

# # print gene overlap size
# with open(model_dir / "vocab.json", "r") as f:
#     vocab_dict = json.load(f)
# print("(pre HVG) Gene overlap size:", len(set(adata.var[gene_col]).intersection(set(vocab_dict.keys()))), "out of", len(adata.var))

# save the var
# adata.var["found_in_scgpt"] = adata.var[gene_col].isin(vocab_dict.keys())
# # adata.var.to_csv(save_dir / "scgpt_gene_found.csv")

# adata = adata[:, adata.var["found_in_scgpt"]]

# # filter genes
# sc.pp.filter_genes(adata, min_cells=10)
# print("adata shape: ", adata.shape)


# preprocess
# sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3', batch_key='study_name')
# adata.var[adata.var['highly_variable']].to_csv(save_dir / "scgpt_scref_3k_hvg_genes.csv")
# print("saved HVG genes")
# adata = adata[:, adata.var['highly_variable']]
hvg_df = pd.read_csv(save_dir / "scgpt_scref_3k_hvg_genes.csv")


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata = adata[:, adata.var["gene"].isin(hvg_df["gene"])]
print("adata shape: ", adata.shape)

# print gene overlap size
with open(model_dir / "vocab.json", "r") as f:
    vocab_dict = json.load(f)
print("(post HVG) Gene overlap size:", len(set(adata.var[gene_col]).intersection(set(vocab_dict.keys()))), "out of", len(adata.var))

default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)

# embed the data
adata = scg.tasks.embed_data(
            adata,
            model_dir,
            gene_col=gene_col,
            batch_size=64,
        )

# save the embeddings to a csv
obs_df = adata.obs[['soma_joinid', 'barcode', 'study_name']].copy()
emb_df = adata.obsm['X_scGPT'].copy()
emb_df = pd.DataFrame(emb_df, columns=[f"embedding_{i}" for i in range(emb_df.shape[1])])
print(obs_df.head())
print(emb_df.head())

emb_df = pd.merge(obs_df, emb_df, on=obs_df.index)

print("- emb_df SHAPE: ", emb_df.shape)

# save the embeddings
os.makedirs(save_dir, exist_ok=True)
emb_df.to_csv(save_dir / "scgpt_zero_shot_embedding.csv", index=False)

