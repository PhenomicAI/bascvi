import os
import scanpy 
import numpy as np
import pandas as pd
import pickle as pkl
import umap



embeddings = np.loadtxt('./prod_logs/embeddings.csv',delimiter=',')
loc = embeddings[:,-3:].astype(int)
embeddings = embeddings[:,:-3]

print('Running UMAP on embeddings...')

umap_projections = umap.UMAP().fit_transform(embeddings)

print('Concatentating Obs...')

with open('./prod_logs/file_list.list', 'rb') as handle:
    file_list = pkl.load(handle)


file_inds = loc[:,0] + loc[:,1]
embedding_cols = ['embedding_' + str(i) for i in range(10)]

all_obs = []

for i,f in enumerate(file_list):
    
    mask = file_inds == i
    
    
    #adata_name = f.split('/')[-1][:-5] #adata mode
    
    adata_name = '_'.join(f.split('/')[:-1])

    obs_ = pd.read_csv('./prod_logs/masked_adatas/' + adata_name + '.tsv',delimiter='\t')

    if obs_.shape[0] > np.sum(mask):
        print('Warning file longer than mask:  ', f, 'by : ', obs_.shape[0]-np.sum(mask))
        cut = np.sum(mask)
        obs_ = obs_[:cut]
        obs_[embedding_cols] = embeddings[mask]
        obs_['umap_0'] = umap_projections[:,0][mask]
        obs_['umap_1'] = umap_projections[:,1][mask]
        
    else:
        obs_[embedding_cols] = embeddings[mask]
        obs_['umap_0'] = umap_projections[:,0][mask]
        obs_['umap_1'] = umap_projections[:,1][mask]
    
    all_obs.append(obs_)

embeddings_concat = pd.concat(all_obs)

print('Writing data...')
#embeddings_concat = embeddings_concat.sort_values(['study_name','disease_name','sample_name'])
embeddings_concat.to_csv('./prod_logs/embedded_cell_data.tsv',sep='\t')

print('Finished')

