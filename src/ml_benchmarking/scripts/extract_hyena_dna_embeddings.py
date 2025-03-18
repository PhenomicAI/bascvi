import numpy as np
import pandas as pd
import pickle
import random
import os
import pandas as pd
import tiledbsoma as soma
import tiledb
from dotenv import load_dotenv

# GENE LIST AND FASTA TO SUB FASTA

# with open('/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt', 'r') as f:
#     gene_list = f.readlines()
#     gene_list = [g.strip() for g in gene_list]

# # open text file and read every other line
# with open('/home/ubuntu/paper_repo/bascvi/data/Ensembl-Human-Sequences.txt', 'r') as f:
#     lines = f.readlines()

# # extract the names of the hyenas
# ensembl_gene_names = [line.strip().split(' ')[1] for i, line in enumerate(lines) if i % 2 == 1]
# ensembl_gene_names = list(filter(lambda n: n.strip() != '', ensembl_gene_names))

# # ensembl gene names to symbol
# gene_map_df = pd.read_csv('/home/ubuntu/paper_repo/bascvi/data/HPA_ENSEMBL_PHENOMIC.merged_versions.tsv.gz', delimiter='\t')

# gene_map_df["ensembl_combined"] = gene_map_df["Ensembl_main"]

# not_found = ~gene_map_df["ensembl_combined"].isin(ensembl_gene_names)

# for i in range(gene_map_df.shape[0]):
#     if not_found[i]:
#         for alt in gene_map_df.loc[i, "Ensembl_alternatives"].split(";"):
#             if alt in ensembl_gene_names:
#                 gene_map_df.loc[i, "ensembl_combined"] = alt
#                 break

# # subset gene map to gene_list
# gene_map_df = gene_map_df[gene_map_df["GeneName_main"].str.lower().isin(gene_list)]

# ensembl_to_save = gene_map_df["ensembl_combined"].values.tolist()
# output_txt = ""
# count = 0
# # open text file and read every other line
# with open('/home/ubuntu/paper_repo/bascvi/data/Ensembl-Human-Sequences.txt', 'r') as f:
#     while True:
#         line1 = f.readline()
#         if not line1:
#             break
#         if line1.strip() == '':
#             continue
#         line2 = f.readline()
#         if not line2:
#             break


#         if line1.strip().split(' ')[1] in ensembl_to_save:
#             output_txt += line1 + line2
#             count += 1
#             print(count, line1.strip()[:5], line2.strip()[:5])


# print(count, len(gene_list))
# # save ensembl gene names
# with open('/home/ubuntu/paper_repo/bascvi/data/human_mouse_seq_ensembl_combined.txt', 'w') as f:
#     f.write(output_txt)




# EMBEDDINGS TO DICT

# open text file and read every other line
ensembl_gene_names = []
with open('/home/ubuntu/paper_repo/bascvi/data/human_mouse_seq_ensembl_combined.txt', 'r') as f:
    while True:
        line1 = f.readline()
        if not line1:
            break
        if line1.strip() == "":
            continue
        line2 = f.readline()
        if not line2:
            break

        if line2.strip() == "":
            continue

        ensembl_gene_names.append(line1.strip().split(' ')[1])


# load embs
embs = np.load('/home/ubuntu/paper_repo/bascvi/data/hyena_160k_human_mouse_embeddings_rolling_mean.npy')

# make df with gene names and embeddings
emb_df = pd.DataFrame(embs, index=ensembl_gene_names, columns=[f"embedding_{i}" for i in range(embs.shape[1])])


# ensembl gene names to symbol
gene_map_df = pd.read_csv('/home/ubuntu/paper_repo/bascvi/data/HPA_ENSEMBL_PHENOMIC.merged_versions.tsv.gz', delimiter='\t')
gene_map_df["ensembl_combined"] = gene_map_df["Ensembl_main"]
not_found = ~gene_map_df["ensembl_combined"].isin(emb_df.index)

for i in range(gene_map_df.shape[0]):
    if not_found[i]:
        for alt in gene_map_df.loc[i, "Ensembl_alternatives"].split(";"):
            if alt in ensembl_gene_names:
                gene_map_df.loc[i, "ensembl_combined"] = alt
                break

gene_map_df.set_index('ensembl_combined', inplace=True)
#gene_map_df.loc[ensembl_gene_names, [f"embedding_{i}" for i in range(embs.shape[1])]] = embs
gene_map_df = gene_map_df.join(emb_df, how='inner')

gene_map_df = gene_map_df[["GeneName_main"] + [f"embedding_{i}" for i in range(embs.shape[1])]]

# group by gene name and take the mean of the embeddings
gene_map_df = gene_map_df.groupby('GeneName_main').mean()

# turn into a dictionary
gene_name = gene_map_df.index
embeddings = gene_map_df.values

gene_name_to_embedding = dict(zip(gene_name, embeddings))

# save the dictionary
import torch
torch.save(gene_name_to_embedding, '/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/HyenaDNA/human_embedding.torch')


