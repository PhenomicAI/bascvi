import anndata as ad
import pandas as pd

# Path to the .zarr directory (can be local or remote like S3, GCS, etc.)
zarr_path = "/home/ubuntu/scREF_test/data/scref_ICLR_2025/zarr_train_blocks/shuffled_fragment_0000_0000.zarr"

# Load the AnnData object from Zarr
adata = ad.read_zarr(zarr_path)

# Print basic info
print(f"Loaded AnnData object with shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var columns: {list(adata.var.columns)}")

# Access obs and var DataFrames
obs_df = adata.obs
var_df = adata.var

# Example: display first few rows
print("\nFirst 5 rows of obs:")
print(obs_df.head())

print("\nFirst 5 rows of var:")
print(var_df.head())