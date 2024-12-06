from bascvi.utils.protein_embeddings import get_centroid_distance_matrix
from bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from bascvi.datamodule.soma.datamodule import TileDBSomaIterDataModule

import json

import numpy as np

# Base config
base_config_path = "/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/templates/macrogenes/train_multispecies.json"
with open(base_config_path, 'r') as file:
    config = json.load(file)

# Set up datamodule
datamodule = TileDBSomaIterDataModule(root_dir=".", **config["datamodule"]["options"])
datamodule.setup()

# Protein embeddings directory
protein_embeddings_dir = "/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/ESM2_phenomic"

# Get centroid distance matrix
matrix = get_centroid_distance_matrix(protein_embeddings_dir, gene_list=datamodule.gene_list, species_list=['human', 'mouse', 'lemur', 'macaque', 'rat'], num_clusters=10000)

# Save matrix
matrix_path = "/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/ESM2/10k_centroid_distance_matrix_multispecies_06Nov2024.npy"
np.save(matrix_path, matrix)