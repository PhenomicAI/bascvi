import numpy as np
import torch

example_embs = torch.load("/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/ESM2/human_embedding.torch")

emb_keys = list(example_embs.keys())

# Generate random gene embeddings
emb_dim = 5120

random_embs = {}
for key in emb_keys:
    random_embs[key] = torch.randn(emb_dim)

torch.save(random_embs, "/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/random/human_embedding.torch")



