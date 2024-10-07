import pandas as pd
import pickle
import tiledbsoma as soma
import tiledb 
from dotenv import load_dotenv
import os
import torch
import numpy as np
import random

def get_protein_embeddings(protein_embeddings_dir, species_list):
    protein_embeddings_dict = {}
    for species in species_list:
        protein_embeddings_dict[species] = torch.load(os.path.join(protein_embeddings_dir, species + "_embedding.torch"))
    return protein_embeddings_dict

load_dotenv("/home/ubuntu/.aws.env")

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SOMA_CORPUS_URI = "s3://pai-scrnaseq/sctx_gui/corpora/combined_scref_no_ortho/"

soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

var_df = soma_experiment.ms['RNA'].var.read().concat().to_pandas()
var_df
prot_emb_dict = get_protein_embeddings("/home/ubuntu/saturn/eval_scref_plus_mu/protein_embeddings_export/ESM2", ["human", "mouse"])

genes_in_soma = var_df.gene.tolist()

genes_to_use = []
for s in prot_emb_dict.keys():
    genes_in_common = set([k.lower() for k in prot_emb_dict[s].keys()]).intersection(set([g.lower() for g in genes_in_soma]))
    genes_to_use.extend(list(genes_in_common))
    print(f"{s}: {len(set([k.lower() for k in prot_emb_dict[s].keys()]).intersection(set([g.lower() for g in genes_in_soma])))} genes in soma out of {len(prot_emb_dict[s].keys())} in protein embeddings")
print(f"Total genes to use: {len(genes_to_use)}")

# save the genes to use as text
with open("/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt", "w") as f:
    f.write("\n".join(genes_to_use))
