import os
import scanpy
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import gc
import pandas as pd

DATA_DIR = './data/'
TEMP_DIR = './reference_data/temp_store/'
file_paths = glob.glob(os.path.join(DATA_DIR, "*.h5ad"))

# Compile: 
#   1) List of all genes and frequency in files
#   2) List of all barcodes


gene_ref_calc = False

if gene_ref_calc:
    gene_dict = {}

all_barcodes = []

for f in file_paths:
    print('Reading genes: ', f)
    adata_backed = scanpy.read(f,backed='r')
    
    if gene_ref_calc:
        gene_list_ = adata_backed.var['gene'].values
        for g in gene_list_:
            if g in gene_dict:
                gene_dict[g] += 1
            else:
                gene_dict[g] = 1

    all_barcodes += list(adata_backed.obs.index.values)

# Specify reference gene list

#reference_gene_list = [k for k,v in gene_dict.items() if v>5]

# Create temp folder for saving adata blocks

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

if gene_ref_calc:

    # Save reference gene list

    with open(TEMP_DIR + 'reference_genes.list', 'wb') as handle:
        pkl.dump(reference_gene_list, handle, protocol=pkl.HIGHEST_PROTOCOL) 

reference_gene_list = pd.read_csv('gene_list_30.txt').index.values

print('Reference gene number = ', len(reference_gene_list))

def log_mean(g, X):
    
    vals = X[g.values,:]
    log_counts = np.log(vals.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(g, X):
    
    vals = X[g.values,:]
    log_counts = np.log(vals.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var

batch_keys = ['sample_name','study_name','scrnaseq_protocol']

b_size = 5000 # Block size parameter for random chunking of adatas
n_cells = len(all_barcodes)
block_n = n_cells//b_size

random_inds = np.random.permutation(np.arange(n_cells))
block_mapping = {all_barcodes[random_inds[i]]:i//b_size for i in range(n_cells)}

print('Splitting data into :' , block_n , ' blocks')

# Chunk adatas into blocks - load in n adata (nad) at once to save on file write number

nad = 5

batch_counters = {b:0 for b in batch_keys}
split_paths = []

binary_masks = {}

for ii in range(0,len(file_paths),nad):
    adatas = []
    
    for jj in range(nad):
        if ii+jj < len(file_paths):
            
            f_path = file_paths[ii+jj]
            
            print('Reading File: ', ii+jj, ' out of ', len(file_paths), ' file name is ', f_path)            
            adata_ = scanpy.read(f_path)
            
            if not adata_.X.dtype == np.float32:
                adata_.X = adata_.X.astype(np.float32)

            ref = scanpy.AnnData(X=np.zeros((1,len(reference_gene_list)),dtype=np.float32),var={'gene':reference_gene_list})
            ref.var = ref.var.set_index(ref.var['gene'])

            
            var_gene_set = adata_.var['gene']
            mask = np.asarray([r in var_gene_set for r in reference_gene_list])
            
            print('Gene number in ref: ', np.sum(mask), 'Adata shape', adata_.shape)
            
            binary_masks[adata_.obs['study_name'].values[0]] = mask
            
            adata_ = scanpy.concat([ref,adata_],join='outer')
            adata_ = adata_[:,reference_gene_list]

            obs_ = adata_.obs.copy()
            
            if 'age_range_years' in obs_:
                obs_['age_range_years'] = obs_['age_range_years'].astype(str)
            else:
                obs_['age_range_years'] = np.zeros(obs_.shape[0],).astype(str)
            
            adata_.obs = obs_
            
            #Filter cells with very few gene reads
            
            gene_counts = adata_.X.getnnz(axis=1)
            mask = gene_counts > 300
            adata_ = adata_[mask,:].copy()
            
            #Batch calcs - assign batch ID that's unique across all datasets

            for i,b in enumerate(batch_keys):
                batch_id = "batch_" + str(i+1)
                codes = adata_.obs[b].astype("category").cat.codes.astype(int)
                adata_.obs[batch_id] = codes + batch_counters[b]
                batch_counters[b] += codes.max() + 1
                
            #Add local_l_mean_key and local_l_var_key to adata.obs
            
            adata_.obs['int_index'] = list(range(adata_.shape[0]))
            
            for i in range(len(batch_keys)):

                header_m = "l_mean_batch_" + str(i+1)
                adata_.obs[header_m] = adata_.obs.groupby(batch_keys[i])["int_index"].transform(log_mean, adata_.X)
                header_v = "l_var_batch_" + str(i+1)
                adata_.obs[header_v] = adata_.obs.groupby(batch_keys[i])["int_index"].transform(log_var, adata_.X)
            
            adatas.append(adata_)
    
    adata_chunk = scanpy.concat(adatas, join="inner", index_unique=None)
    adata_chunk.obs['block'] = adata_chunk.obs['barcode'].apply(lambda x : block_mapping[x])

    print('Writing split : ', str(ii//nad).zfill(4))
    for i in range(block_n):
        
        split_path = TEMP_DIR + 'chunk_' + str(ii//nad).zfill(4) + '_split_' + str(i).zfill(4) + '.h5ad'
        
        adata_split = adata_chunk[adata_chunk.obs['block']==i,:].copy()
        adata_split.write(split_path)
        split_paths.append(split_path)
    
    del adatas
    gc.collect()

#Save max batch val for one hot vector calcs

batch_dict = {b:batch_counters[b] for b in batch_keys}

print(batch_dict)

with open(TEMP_DIR+'bdict.dict', "wb") as fh:
    pkl.dump([batch_dict, batch_keys], fh)


with open(TEMP_DIR+'binary_masks.dict', "wb") as fh:
    pkl.dump(binary_masks, fh)

print('Consolidating split blocks...')

file_lengths = {}

for i in range(block_n):
    print('Block: ',i)
    adatas = []
    for split_path in split_paths: 
        if str(i).zfill(4) in split_path[-12:]:
            adata_ = scanpy.read(split_path)
            adatas.append(adata_)
            os.remove(split_path)            

    fname_ = TEMP_DIR+'adata_block_'+str(i).zfill(4)+'.h5ad'
    
    adata_block = scanpy.concat(adatas, join="inner", index_unique=None)
    adata_block = adata_block[np.random.permutation(adata_block.shape[0]),:].copy()
    adata_block.write(fname_)
    file_lengths[fname_] = adata_block.shape[0]

    del adatas
    del adata_block
    gc.collect()

with open(TEMP_DIR+'file_lengths.dict', "wb") as fh:
    pkl.dump(file_lengths, fh)



