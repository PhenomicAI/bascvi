import pandas as pd
import numpy as np
import os
import torch
from sklearn.cluster import KMeans

from scipy.stats import rankdata



def get_protein_embeddings(protein_embeddings_dir, species_list):
    protein_embeddings_dict = {}
    for species in species_list:
        protein_embeddings_dict[species] = torch.load(os.path.join(protein_embeddings_dir, species + "_embedding.torch"))
    return protein_embeddings_dict


# STACKED PROTEIN EMBEDDINGS MATRIX

def get_stacked_protein_embeddings_matrix(protein_embeddings_dir, species_list, gene_list):
    # init list for protein embeddings
    protein_embeddings_matrix = []

    # get protein embeddings given species list
    protein_embeddings_dict = get_protein_embeddings(protein_embeddings_dir, species_list)

    # get embeddings for genes in gene list
    genes_with_embeddings = {g.lower(): emb for s in protein_embeddings_dict.keys() for g, emb in protein_embeddings_dict[s].items() if g.lower() in gene_list}

    # get zero embedding for genes not found in embeddings
    zero_embedding = np.zeros(genes_with_embeddings[list(genes_with_embeddings.keys())[0]].shape)

    # stack embeddings in order of gene list
    protein_embeddings_matrix = [genes_with_embeddings[g].cpu().numpy() if g.lower() in genes_with_embeddings else zero_embedding for g in gene_list]

    return np.array(protein_embeddings_matrix)


# CENTROID DISTANCE MATRIX (inspired by SATURN)

def get_centroid_distance_matrix(protein_embeddings_dir, species_list, gene_list, num_clusters=4000, seed=42, pre_normalize=False, score_function="default"):
    # get protein embeddings matrix
    protein_embeddings_matrix = get_stacked_protein_embeddings_matrix(protein_embeddings_dir, species_list, gene_list)
    
    # normalize embeddings
    if pre_normalize:
        protein_embeddings_matrix = protein_embeddings_matrix / np.sum(protein_embeddings_matrix, axis=1, keepdims=True)

    # run kmeans
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=seed, verbose=1)

    # fit kmeans model
    print("Fitting KMeans model...")
    kmeans_model = kmeans_model.fit(protein_embeddings_matrix)

    # get distance from each gene to each cluster centroid
    print("Transforming protein embeddings...")
    distance_matrix = kmeans_model.transform(protein_embeddings_matrix)
    
    # get scores for each gene
    print("Getting scores...")
    if score_function == "default":
        return default_centroids_scores(distance_matrix)
    elif score_function == "one_hot":
        return one_hot_centroids_scores(distance_matrix)
    elif score_function == "smoothed":
        return smoothed_centroids_score(distance_matrix)


def default_centroids_scores(distance_matrix):
    """
    Convert KMeans distances to centroids to scores.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(distance_matrix, axis=1) # rank 1 is close rank 2000 is far

    to_scores = np.log1p(1 / ranked) # log 1 is close log 1/2000 is far

    to_scores = ((to_scores) ** 2)  * 2
    return to_scores


def one_hot_centroids_scores(distance_matrix):
    """
    Convert KMeans distances to centroids to scores. All or nothing, so closest centroid has score 1, others have score 0.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(distance_matrix, axis=1) # rank 1 is close rank 2000 is far
    
    to_scores = (ranked == 1).astype(float) # true, which is rank 1, is highest, everything else is 0
    return to_scores


def smoothed_centroids_score(distance_matrix):
    """
    Convert KMeans distances to centroids to scores. Smoothed version of original function, so later ranks have larger values.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(distance_matrix, axis=1) # rank 1 is close rank 2000 is far
    to_scores = 1 / ranked # 1/1 is highest, 1/2 is higher than before, etc.
    return to_scores