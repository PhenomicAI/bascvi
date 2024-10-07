import pandas as pd
import numpy as np
import os
import torch


def get_protein_embeddings(protein_embeddings_dir, species_list):
    protein_embeddings_dict = {}
    for species in species_list:
        protein_embeddings_dict[species] = torch.load(os.path.join(protein_embeddings_dir, species + "_embedding.torch"))
    return protein_embeddings_dict


# STACKED PROTEIN EMBEDDINGS MATRIX
def get_stacked_protein_embeddings_matrix(protein_embeddings_dir, species_list, gene_list):
    protein_embeddings_matrix = []

    protein_embeddings_dict = get_protein_embeddings(protein_embeddings_dir, species_list)

    genes_with_embeddings = {g.lower(): emb for s in protein_embeddings_dict.keys() for g, emb in protein_embeddings_dict[s].items() if g.lower() in gene_list}

    zero_embedding = np.zeros(genes_with_embeddings[list(genes_with_embeddings.keys())[0]].shape)
    protein_embeddings_matrix = [genes_with_embeddings[g].cpu().numpy() if g.lower() in genes_with_embeddings else zero_embedding for g in gene_list]
    return np.array(protein_embeddings_matrix)


# CENTROID DISTANCE MATRIX


