import anndata as ad
import scanpy as sc
import zarr
import os
import glob
from tqdm import tqdm


def convert_h5ad_to_zarr(h5ad_path, output_dir):
    """Convert a single h5ad file to zarr format."""
    print(f"Processing: {os.path.basename(h5ad_path)}")
    
    # Load the h5ad file
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    print(adata.obs.columns)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.zarr")
    
    # Save to zarr format
    adata.write_zarr(output_path)
    print(f"  Saved to: {output_path}")
    
    return output_path


def main():
    # Define paths
    input_dir = "/home/ubuntu/scRNA/scMARK/scmark"
    output_dir = "/home/ubuntu/scRNA/scMARK/zarr"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all h5ad files
    h5ad_files = glob.glob(os.path.join(input_dir, "*.h5ad"))
    h5ad_files.sort()
    
    print(f"Found {len(h5ad_files)} h5ad files to convert")
    
    # Convert each file
    converted_files = []
    
    for h5ad_path in tqdm(h5ad_files, desc="Converting files"):
        output_path = convert_h5ad_to_zarr(h5ad_path, output_dir)
        converted_files.append(output_path)
    
    print(f"\nConversion complete! Converted {len(converted_files)} files to: {output_dir}")
    print("Files are ready for the shuffle zarr script.")


if __name__ == "__main__":
    main()