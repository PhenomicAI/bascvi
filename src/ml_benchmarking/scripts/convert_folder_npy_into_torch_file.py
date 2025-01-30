import os
import numpy as np
import torch

def load_npy_files_to_dict(folder_path):
    """
    Load all .npy files from a folder into a dictionary.

    Args:
        folder_path (str): Path to the folder containing .npy files.

    Returns:
        dict: A dictionary where keys are file names (without extension)
              and values are the arrays loaded from the .npy files.
    """
    npy_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            key = os.path.splitext(file_name)[0]  # Remove .npy extension
            emb = np.load(file_path)
            # take mean of embeddings
            npy_dict[key] = np.mean(emb, axis=0)
    return npy_dict

def save_dict_as_torch(dict_obj, output_file):
    """
    Save a dictionary as a .torch file.

    Args:
        dict_obj (dict): The dictionary to save.
        output_file (str): The output .torch file path.
    """
    torch.save(dict_obj, output_file)

if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Convert a folder of .npy files to a .torch file.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing .npy files.")
    parser.add_argument("output_file", type=str, help="Path to save the .torch file.")
    args = parser.parse_args()

    # Load .npy files and save as .torch file
    npy_dict = load_npy_files_to_dict(args.folder_path)
    save_dict_as_torch(npy_dict, args.output_file)

    print(f"Saved dictionary to {args.output_file}")
