import anndata as ad
import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import zarr
from tqdm import tqdm
import gc
import hashlib

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

def generate_cell_idx_with_counter(study_name, barcode, study_idx, sample_idx, cell_counter):
    """
    Generate a globally unique cell identifier using a monotonically increasing counter.
    
    This approach combines a hash of the cell identifiers with a monotonically increasing
    counter to ensure global uniqueness without collision risk. The hash provides
    deterministic behavior while the counter ensures uniqueness.
    
    Parameters
    ----------
    study_name : str
        Name of the study
    barcode : str
        Cell barcode
    study_idx : int
        Study index
    sample_idx : int
        Sample index
    cell_counter : int
        Monotonically increasing counter for this cell
    
    Returns
    -------
    int
        Globally unique cell identifier (fits within int64 range)
    """
    # Create a unique string combining all identifiers
    unique_string = f"{study_name}_{barcode}_{study_idx}_{sample_idx}"
    # Generate hash and convert to integer
    hash_object = hashlib.md5(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    # Use first 8 characters of hash (32-bit) and combine with counter
    hash_part = int(hash_hex[:8], 16)
    # Combine hash with counter to create unique identifier
    # Use modulo to ensure result fits in int64 range
    cell_idx = ((hash_part * 1000000) + cell_counter) % (2**63)
    return cell_idx

def generate_cell_idx(study_name, barcode, study_idx, sample_idx):
    """
    Generate a globally unique cell identifier based on study, barcode, and indices.
    
    This function creates a deterministic, globally unique identifier for each cell
    by combining the study name, cell barcode, study index, and sample index into
    a hash. This ensures that the same cell will always get the same cell_idx,
    even across different processing runs or data shuffling operations.
    
    The hash is truncated to fit within int64 range to avoid overflow issues.
    
    Parameters
    ----------
    study_name : str
        Name of the study
    barcode : str
        Cell barcode
    study_idx : int
        Study index
    sample_idx : int
        Sample index
    
    Returns
    -------
    int
        Globally unique cell identifier (fits within int64 range)
    """
    # Create a unique string combining all identifiers
    unique_string = f"{study_name}_{barcode}_{study_idx}_{sample_idx}"
    # Generate hash and convert to integer
    hash_object = hashlib.md5(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    # Convert first 12 characters of hash to integer (48-bit) to fit within int64 range
    # This gives us 2^48 possible values, which is more than enough for cell identification
    cell_idx = int(hash_hex[:12], 16)
    return cell_idx

def load_obs_column(zarr_obs_group, key):
    group = zarr_obs_group[key]
    attrs = dict(group.attrs)

    if attrs.get("encoding-type") == "categorical":
        # Load codes and categories
        codes = group['codes'][:]
        categories = group['categories'][:]
        
        # Decode bytes to strings if necessary
        if np.issubdtype(categories.dtype, np.bytes_):
            categories = np.char.decode(categories, 'utf-8')

        return categories[codes]
    
    else:
        # Fallback to regular dataset
        return group[:]


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

def fragment_zarr(input_dir, fragment_dir, output_dir, gene_list, target_shuffle_size=50000):

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

    if not os.path.exists(fragment_dir):
        os.makedirs(fragment_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sample_counter = 0
    z_counter = 0
    global_cell_counter = 0  # Global counter for unique cell identification

    for zarr_path in input_zarr_files:

        study_name = zarr_path.split('/')[-1].split('.')[0]

        # Load the Zarr object
        z = zarr.open_group(zarr_path, mode='r')
        full_shape = z['X'].attrs['shape']

        # Read gene names from Zarr file
        zarr_genes = z['var']['gene'][:].tolist()

        print(f"Processing {study_name}  with shape: {full_shape[0]} sample count: {sample_counter} zarr counter: {z_counter} gene count: {len(zarr_genes)}")

        # Build index map from gene_list -> zarr index or None

        zarr_gene_to_idx = {gene: i for i, gene in enumerate(zarr_genes)}
        gene_to_zarr_idx = {gene: zarr_gene_to_idx.get(gene, None) for gene in gene_list}
        
        var_df = pd.DataFrame(gene_list, columns=['gene'])

        # extract obs columns in the specified row range

        obs_dict = {}
        obs_group = z['obs']

        for k in obs_group.keys():
            try:
                print(f"Processing key: {k}")
                obs_dict[k] = load_obs_column(obs_group, k)
            except Exception as e:
                print(f"Error loading {k}: {e}")

        # Build dataframe
        obs = pd.DataFrame(obs_dict)

        ad_obs_list = []

        # Determine how to iterate through samples
        if 'sample_name' in obs.columns:
            # Use sample_name for unique identification within this file
            unique_samples = obs['sample_name'].unique()
            sample_iter = tqdm(unique_samples, desc=f"Samples in {study_name}")
            
            for sample_name in sample_iter:
                mask = obs['sample_name'] == sample_name
                
                # Convert mask to integer indices
                row_indices = np.where(mask)[0]
                
                sample_X = load_sparse_rows_from_zarr(z['X'], row_indices)   
                sample_obs = obs[mask].copy()

                sample_obs['study_name'] = study_name
                sample_obs['study_idx'] = z_counter

                # Assign unique sample_idx for this sample (same sample_name in different files gets different IDs)
                sample_obs['sample_idx'] = sample_counter
                
                # Generate globally unique cell_idx for each cell using hash + counter approach
                # This ensures each cell has a unique, deterministic identifier that can be used
                # to map model outputs back to the original input cells
                sample_obs['cell_idx'] = sample_obs.apply(
                    lambda row: generate_cell_idx_with_counter(
                        study_name, 
                        str(row['barcode']), 
                        z_counter, 
                        sample_counter,
                        global_cell_counter + row.name  # Use row index for counter
                    ), axis=1
                )
                
                # Update global cell counter
                global_cell_counter += len(sample_obs)
                sample_counter += 1

                ad_obs_list.append(sample_obs)
                # Clean up per-sample objects
                del sample_X, sample_obs
                gc.collect()
        else:
            # Fallback: treat entire file as one sample if no sample_name or sample_idx column
            print(f"No sample_name column found in {study_name}, treating entire file as one sample")
            
            # Add required columns to the original obs dataframe
            # Load all data to calculate log mean and variance
            all_X = load_sparse_rows_from_zarr(z['X'], np.arange(len(obs)))
            obs['study_name'] = study_name
            obs['study_idx'] = z_counter
            obs['sample_idx'] = sample_counter
            
            # Generate globally unique cell_idx for each cell using hash + counter approach
            # This ensures each cell has a unique, deterministic identifier that can be used
            # to map model outputs back to the original input cells
            obs['cell_idx'] = obs.apply(
                lambda row: generate_cell_idx_with_counter(
                    study_name, 
                    str(row['barcode']), 
                    z_counter, 
                    sample_counter,
                    global_cell_counter + row.name  # Use row index for counter
                ), axis=1
            )
            
            # Update global cell counter
            global_cell_counter += len(obs)
            sample_counter += 1

            # Use the modified obs dataframe directly
            obs_df = obs.copy()
            # Clean up per-sample objects
            gc.collect()

        z_counter += 1

        # Only concatenate if we processed multiple samples
        if 'sample_name' in obs.columns:
            obs_df = pd.concat(ad_obs_list)
            # Clean up ad_obs_list
            del ad_obs_list
        # obs_df is already set in the fallback case
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

            # Filter cells (rows) with at least 300 non-zero gene values
            gene_counts_per_cell = X_final_block.getnnz(axis=1)  # Number of non-zeros per row
            cell_mask = gene_counts_per_cell >= 300

            # Skip block if no cells pass the filter
            if cell_mask.sum() == 0:
                print(f"Skipping block [{start}:{stop}]: no cells with ≥300 expressed genes.")
                continue

            # Apply mask to matrix and obs
            X_final_block_filtered = X_final_block[cell_mask, :]
            obs_filtered = obs_df[start:stop].iloc[cell_mask]

            # Calculate log_mean and log_var from filtered data
            if hasattr(X_final_block_filtered, 'getnnz'):
                log_counts = np.log(X_final_block_filtered.sum(axis=1).A1 + 1e-6)
            else:
                log_counts = np.log(X_final_block_filtered.sum(axis=1) + 1e-6)
            
            obs_filtered['log_mean'] = np.mean(log_counts)
            obs_filtered['log_var'] = np.var(log_counts)

            # Write AnnData object
            ad_write = ad.AnnData(X=X_final_block_filtered, obs=obs_filtered, var=var_df)
            
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

    max_sample_idx = 0
    max_study_idx = 0

    for block_num in sorted(block_map.keys()):
        print(f"\n🔄 Processing block {block_num} with {len(block_map[block_num])} fragments...")
        
        # Load and combine fragments
        adatas = [ad.read_zarr(path) for path in tqdm(sorted(block_map[block_num]))]
        combined = ad.concat(adatas, axis=0, merge="same", index_unique=None)

        max_sample_idx_block = max(max_sample_idx, combined.obs['sample_idx'].max())
        if max_sample_idx_block > max_sample_idx:
            max_sample_idx = max_sample_idx_block
        
        max_study_idx_block = max(max_study_idx, combined.obs['study_idx'].max())
        if max_study_idx_block > max_study_idx:
            max_study_idx = max_study_idx_block

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
    print(f"Max sample idx: {max_sample_idx}")
    print(f"Max study idx: {max_study_idx}")


def generate_feature_presence_matrix(input_dir, gene_list, output_dir):
    """
    Generate a feature presence matrix for each study (study_idx/z_counter) and gene in gene_list.
    The matrix is shape (num_studies, num_genes), with 1 if the gene is present in the study, 0 otherwise.
    Saves the matrix as 'feature_presence_matrix.npy' in the output directory.
    """
    import numpy as np
    import zarr
    import os

    # Find all .zarr files in the input_dir
    zarr_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.zarr')]
    zarr_files.sort()
    
    num_studies = len(zarr_files)
    num_genes = len(gene_list)
    feature_presence = np.zeros((num_studies, num_genes), dtype=np.int8)

    for z_counter, zarr_path in enumerate(zarr_files):
        z = zarr.open_group(zarr_path, mode='r')
        zarr_genes = z['var']['gene'][:].tolist()
        zarr_gene_set = set(zarr_genes)
        for g_idx, gene in enumerate(gene_list):
            if gene in zarr_gene_set:
                feature_presence[z_counter, g_idx] = 1
    
    np.save(os.path.join(output_dir, 'feature_presence_matrix.npy'), feature_presence)
    print(f"Feature presence matrix saved to {os.path.join(output_dir, 'feature_presence_matrix.npy')}")


input_dir = "/home/ubuntu/scRNA/scMARK/zarr"
fragment_dir = "/home/ubuntu/scRNA/scMARK/zarr_fragments"
output_dir = "/home/ubuntu/scRNA/scMARK/zarr_train_blocks"

gene_list = pd.read_csv("/home/ubuntu/scRNA/bascvi/src/ml_benchmarking/data/genes_2ksamples_10cells.txt",index_col=0)
gene_list = gene_list.index.tolist()


fragment_zarr(input_dir, fragment_dir, output_dir, gene_list)
shuffle_and_refragment(fragment_dir,output_dir)
generate_feature_presence_matrix(input_dir, gene_list, output_dir)



