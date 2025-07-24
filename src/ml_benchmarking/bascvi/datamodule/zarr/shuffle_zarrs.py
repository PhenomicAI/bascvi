import anndata as ad
import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import zarr
from tqdm import tqdm
import gc

import os
import re
import anndata as ad
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pickle

def log_mean(X):
    """Compute the mean of the log total counts per cell."""
    log_counts = np.log(X.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(X):
    """Compute the variance of the log total counts per cell."""
    log_counts = np.log(X.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var


def load_sparse_rows_from_zarr(x_group, row_indices, max_block_size: int = 4000) -> sp.csr_matrix:
    """
    Load selected rows from Zarr-backed CSR matrix in efficient blocks.
    """
    row_indices = np.sort(np.array(row_indices, dtype=np.int64))
    n_genes = x_group.attrs['shape'][1]

    result_rows = []

    # Process in blocks of consecutive rows
    i = 0
    while i < len(row_indices):
        block_start = row_indices[i]
        block_end = block_start
        while i < len(row_indices) and row_indices[i] == block_end:
            block_end += 1
            i += 1

        # Now block_start:block_end is a contiguous range
        indptr = x_group['indptr'][block_start:block_end + 1]
        start_ptr = indptr[0]
        end_ptr = indptr[-1]

        indices = x_group['indices'][start_ptr:end_ptr]
        data = x_group['data'][start_ptr:end_ptr]

        # Shift indptr
        indptr = indptr - start_ptr

        block = sp.csr_matrix((data, indices, indptr), shape=(block_end - block_start, n_genes))
        result_rows.append(block)

    return sp.vstack(result_rows, format='csr')

def fragment_zarr(input_dir, fragment_dir, output_dir, target_shuffle_size=50000):

    input_zarr_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.zarr') ]
    input_zarr_files.sort()

    total_count = 0

    for z_counter, zarr_path in enumerate(input_zarr_files):
        z = zarr.open_group(zarr_path, mode='r')
        full_shape = z['X'].attrs['shape']
        total_count += full_shape[0]

    fragment_number = total_count // target_shuffle_size

    print(f"Total count: {total_count}")
    print(f"Fragment number: {fragment_number}")

    ### Main code ###

    gene_list = pd.read_csv("/home/ubuntu/scREF_test/bascvi/src/ml_benchmarking/data/genes_2ksamples_10cells.txt",index_col=0)
    gene_list = gene_list.index.tolist()

    if not os.path.exists(fragment_dir):
        os.makedirs(fragment_dir)
    
    sample_counter = 0
    z_counter = 0

    for zarr_path in input_zarr_files:

        study_name = zarr_path.split('/')[-1].split('.')[0]

        # Load the Zarr object
        z = zarr.open_group(zarr_path, mode='r')
        full_shape = z['X'].attrs['shape']

        print(f"Processing {study_name}  with shape: {full_shape[0]} sample count: {sample_counter} zarr counter: {z_counter}")

        # Read gene names from Zarr file
        zarr_genes = z['var']['gene'][:].tolist()
        
        # Build index map from gene_list -> zarr index or None
        zarr_gene_to_idx = {gene: i for i, gene in enumerate(zarr_genes)}
        gene_to_zarr_idx = {gene: zarr_gene_to_idx.get(gene, None) for gene in gene_list}
        var_df = pd.DataFrame(gene_list, columns=['gene'])

        # extract obs columns in the specified row range

        obs_dict = {}
        for k in z['obs'].keys():
            arr = z['obs'][k]
            if hasattr(arr, 'shape'):  # basic check to make sure it's an array
                obs_dict[k] = arr[:]

        # Build dataframe
        obs = pd.DataFrame(obs_dict)

        ad_obs_list = []

        for sample_idx in tqdm(obs['sample_idx'].unique(), desc=f"Samples in {study_name}"):

            mask = obs['sample_idx'] == sample_idx

            # Convert mask to integer indices
            row_indices = np.where(mask)[0]
            
            sample_X = load_sparse_rows_from_zarr(z['X'], row_indices)   
            sample_obs = obs[mask].copy()

            sample_obs['log_mean'] = log_mean(sample_X)
            sample_obs['log_var'] = log_var(sample_X)

            sample_obs['study_name'] = study_name
            sample_obs['study_idx'] = z_counter

            sample_obs['sample_idx'] = sample_counter
            sample_counter += 1

            ad_obs_list.append(sample_obs)
            # Clean up per-sample objects
            del sample_X, sample_obs
            gc.collect()

        z_counter += 1

        obs_df = pd.concat(ad_obs_list)
        # Clean up ad_obs_list
        del ad_obs_list
        gc.collect()

        # Clean up obs 
        if '_index' in obs_df.columns:
            obs_df.drop(columns=['_index'], inplace=True)

        obs_df.set_index('barcode', inplace=True)
        obs_df.index = obs_df.index.astype(str)
        
        obs_df.index.name = 'barcode'
        block_size = 50000

        for zz, block in enumerate(range(0, full_shape[0], block_size)):

            print(f"Processing {study_name} block {zz} of {full_shape[0]//block_size}")

            # Define block range
            start = block
            stop = block + block_size
            if stop > full_shape[0]:
                stop = full_shape[0]

            X_block = load_sparse_rows_from_zarr(z['X'], np.arange(start, stop))   
            X_block = sp.csc_matrix(X_block)

            # Build columns for each gene in the reference list
            cols = []
            for gene in gene_list:
                zarr_idx = gene_to_zarr_idx[gene]
                if zarr_idx is not None:
                    cols.append(X_block[:, zarr_idx])
                else:
                    # Missing gene: insert zero column
                    zero_col = sp.csr_matrix((X_block.shape[0], 1))
                    cols.append(zero_col)

            # Concatenate all gene columns (in order)
            X_final_block = sp.hstack(cols, format='csr')

            ad_write = ad.AnnData(X=X_final_block, obs=obs_df[start:stop], var=var_df)
            
            print("Writing fragments")

            fragment_size = ad_write.shape[0] // fragment_number
            for i in range(fragment_number):

                fragment_path = os.path.join(fragment_dir, f"{study_name}_fragment_{zz}_{i:04d}.zarr")
                frag = ad_write[i*fragment_size:(i+1)*fragment_size, :].copy()
                frag.write_zarr(fragment_path)
                # Clean up fragment
                del frag
                gc.collect()

            # Clean up block-level objects
            del X_block, X_final_block, ad_write, cols
            gc.collect()

        # Clean up per-file objects
        del obs_df, var_df, z, zarr_genes, zarr_gene_to_idx, gene_to_zarr_idx
        gc.collect()
    
    with open(os.path.join(output_dir,"batch_sizes.pkl"),"wb") as f:
        pickle.dump([z_counter,sample_counter], f)

    print(f"Finished fragmenting {z_counter} zarr files")



def shuffle_and_refragment(
    fragment_dir: str,
    output_dir: str,
    max_fragment_size: int = 5000
):
    """
    Consolidate, shuffle, and re-fragment AnnData Zarr fragments.

    Parameters
    ----------
    fragment_dir : str
        Directory containing *_fragment_####.zarr files.
    output_dir : str
        Directory where re-fragmented shuffled output will be written.
    max_fragment_size : int
        Maximum number of cells (rows) per new fragment.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pattern to extract final 4-digit fragment number
    pattern = re.compile(r'_fragment_\d+_(\d{4})\.zarr$')

    # Group fragments by block number
    block_map = defaultdict(list)
    for fname in os.listdir(fragment_dir):
        match = pattern.search(fname)
        if match:
            block_num = int(match.group(1))
            block_map[block_num].append(os.path.join(fragment_dir, fname))

    print(f"Found {len(block_map)} blocks to process.")

    total_written = 0
    for block_num in sorted(block_map.keys()):
        print(f"\n🔄 Processing block {block_num} with {len(block_map[block_num])} fragments...")
        
        # Load and combine fragments
        adatas = [ad.read_zarr(path) for path in tqdm(sorted(block_map[block_num]))]
        combined = ad.concat(adatas, axis=0, merge="same", index_unique=None)

        # Shuffle rows
        n = combined.n_obs
        shuffled_indices = np.random.permutation(n)
        combined = combined[shuffled_indices].copy()
        print(f"  🔀 Shuffled {n} cells")

        # Re-fragment
        for i in range(0, n, max_fragment_size):
            frag = combined[i:i+max_fragment_size].copy()
            frag_path = os.path.join(output_dir, f"shuffled_fragment_{block_num:04d}_{total_written:04d}.zarr")
            frag.write_zarr(frag_path)
            print(f"  ✅ Wrote {frag.n_obs} cells → {frag_path}")
            total_written += 1

    print(f"\n🎉 Finished. Wrote {total_written} shuffled fragments to: {output_dir}")



input_dir = "/home/ubuntu/scREF_test/data/scref_ICLR_2025/zarr"
fragment_dir = "/home/ubuntu/scREF_test/data/scref_ICLR_2025/zarr_fragments"
output_dir = "/home/ubuntu/scREF_test/data/scref_ICLR_2025/zarr_train_blocks"

fragment_zarr(input_dir, fragment_dir, output_dir)

shuffle_and_refragment(fragment_dir,output_dir)



