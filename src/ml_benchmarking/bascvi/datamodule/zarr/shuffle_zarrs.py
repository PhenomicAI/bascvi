import anndata as ad
import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import zarr
from tqdm import tqdm
import gc
import hashlib
import re
from collections import defaultdict
import pickle
import argparse

def get_zarr_shape(zarr_group, array_name='X', debug=False):
    """
    Get the shape of a Zarr array with backward compatibility for different Zarr versions.
    
    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group containing the array
    array_name : str
        Name of the array to get shape for (default: 'X')
    debug : bool
        Whether to print debug information (default: False)
    
    Returns
    -------
    tuple
        Shape of the array
        
    Raises
    ------
    ValueError
        If shape cannot be determined
    """
    try:
        # Try accessing shape from attrs (older Zarr versions)
        shape = zarr_group[array_name].attrs['shape']
        if debug:
            print(f"Got shape from attrs: {shape}")
        return shape
    except (KeyError, AttributeError):
        try:
            # Try accessing shape directly from the array (newer Zarr versions)
            shape = zarr_group[array_name].shape
            if debug:
                print(f"Got shape from array.shape: {shape}")
            return shape
        except AttributeError:
            # Fallback: try to get shape from the array itself
            x_array = zarr_group[array_name]
            if hasattr(x_array, 'shape'):
                shape = x_array.shape
                if debug:
                    print(f"Got shape from array object: {shape}")
                return shape
            else:
                # Last resort: try to infer from data structure
                if hasattr(x_array, 'attrs') and 'shape' in x_array.attrs:
                    shape = x_array.attrs['shape']
                    if debug:
                        print(f"Got shape from array.attrs: {shape}")
                    return shape
                else:
                    raise ValueError(f"Could not determine shape for {array_name} in zarr group")

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
    # Handle different Zarr versions for shape access
    try:
        n_genes = x_group.attrs['shape'][1]
    except (KeyError, AttributeError):
        try:
            n_genes = x_group.shape[1]
        except AttributeError:
            # Fallback: try to get shape from the group itself
            if hasattr(x_group, 'shape'):
                n_genes = x_group.shape[1]
            else:
                raise ValueError("Could not determine number of genes from Zarr group")

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
        full_shape = get_zarr_shape(z, 'X')
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
        
        print(f"Processing {study_name}")

        # Load the Zarr object
        z = zarr.open_group(zarr_path, mode='r')
        full_shape = get_zarr_shape(z, 'X')

        # Read gene names from Zarr file (zarr v3 compatible)
        print(f"Reading gene names from {zarr_path}")
        try:
            # Try zarr v3 syntax first
            zarr_genes = z['var']['gene'][:].tolist()
            print(f"Successfully read {len(zarr_genes)} genes using zarr v3 syntax")
        except Exception as e:
            print(f"Zarr v3 syntax failed: {e}")
            # Fallback to AnnData for zarr v3 files

            adata = ad.read_zarr(zarr_path)
            zarr_genes = adata.var['gene'].tolist()
            print(f"Successfully read {len(zarr_genes)} genes using AnnData")

        print(f"Processing {study_name}  with shape: {full_shape[0]} sample count: {sample_counter} zarr counter: {z_counter} gene count: {len(zarr_genes)}")
        
        # Debug: Show first few genes from both lists
        print(f"First 5 genes from zarr file: {zarr_genes[:5]}")
        print(f"First 5 genes from gene_list: {gene_list[:5]}")
        
        # Check for overlap
        zarr_gene_set = set(zarr_genes)
        gene_list_set = set(gene_list)
        overlap = zarr_gene_set.intersection(gene_list_set)
        print(f"Gene overlap count: {len(overlap)} out of {len(gene_list)} genes in gene_list")

        # Build index map from gene_list -> zarr index or None
        zarr_gene_to_idx = {gene: i for i, gene in enumerate(zarr_genes)}
        gene_to_zarr_idx = {gene: zarr_gene_to_idx.get(gene, None) for gene in gene_list}
        
        # Debug: Show how many genes were successfully mapped
        mapped_count = sum(1 for idx in gene_to_zarr_idx.values() if idx is not None)
        print(f"Successfully mapped {mapped_count} out of {len(gene_list)} genes")
        
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
        obs['order'] = np.arange(len(obs))

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

                # Calculate log_mean and log_var from sample data
                if hasattr(sample_X, 'getnnz'):
                    log_counts = np.log(sample_X.sum(axis=1).A1 + 1e-6)
                else:
                    log_counts = np.log(sample_X.sum(axis=1) + 1e-6)
                
                sample_obs['log_mean'] = np.mean(log_counts)
                sample_obs['log_var'] = np.var(log_counts)
                
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

        obs_df.sort_values(by='order',inplace=True)
        obs_df.drop(columns=['order'],inplace=True)

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
        try:
            # Try zarr v3 syntax first
            zarr_genes = z['var']['gene'][:].tolist()
        except Exception:
            # Fallback to AnnData for zarr v3 files
            adata = ad.read_zarr(zarr_path)
            zarr_genes = adata.var['gene'].tolist()
        
        zarr_gene_set = set(zarr_genes)
        for g_idx, gene in enumerate(gene_list):
            if gene in zarr_gene_set:
                feature_presence[z_counter, g_idx] = 1
    
    np.save(os.path.join(output_dir, 'feature_presence_matrix.npy'), feature_presence)
    print(f"Feature presence matrix saved to {os.path.join(output_dir, 'feature_presence_matrix.npy')}")


def _default_gene_list_path() -> str:
    # Resolve default gene list relative to this file: ../../../data/genes_2ksamples_10cells.txt
    script_dir = os.path.dirname(__file__)
    default_path = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "data", "genes_2ksamples_10cells.txt"))
    return default_path


def main():
    parser = argparse.ArgumentParser(description="Fragment, shuffle, and re-fragment zarr datasets with configurable paths.")
    parser.add_argument("input_dir", type=str, help="Directory containing input .zarr files")
    parser.add_argument("--fragment_dir", type=str, default=None, help="Output directory for initial fragments. Defaults to '<input_dir>_zarr_fragments'")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for shuffled train blocks. Defaults to '<input_dir>_zarr_train_blocks'")
    parser.add_argument("--gene_list", type=str, default=None, help="Path to gene list file (txt). Defaults to data/genes_2ksamples_10cells.txt relative to this script")

    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir_stripped = input_dir[:-1] if input_dir.endswith(os.sep) else input_dir

    fragment_dir = args.fragment_dir if args.fragment_dir is not None else f"{input_dir_stripped}_zarr_fragments"
    output_dir = args.output_dir if args.output_dir is not None else f"{input_dir_stripped}_zarr_train_blocks"

    gene_list_path = args.gene_list if args.gene_list is not None else _default_gene_list_path()
    if not os.path.isfile(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found at: {gene_list_path}")

    gene_list_df = pd.read_csv(gene_list_path, index_col=0, header=None)
    gene_list = gene_list_df.index.tolist()
    print('gene list: ', gene_list[:20])

    fragment_zarr(input_dir, fragment_dir, output_dir, gene_list)
    shuffle_and_refragment(fragment_dir, output_dir)
    generate_feature_presence_matrix(input_dir, gene_list, output_dir)


if __name__ == "__main__":
    main()



