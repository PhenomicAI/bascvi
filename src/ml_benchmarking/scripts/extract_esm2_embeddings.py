import torch
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
import numpy as np
import os
import torch
from tqdm import tqdm 

# Define a function to load sequences from a FASTA file
def load_fasta(fasta_file):
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    return sequence_ids, sequences

# Define a function to retrieve embeddings for a single sequence
def get_embedding(sequence, model, tokenizer, device):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (sequence_length, embedding_dim)
        embeddings = embeddings.mean(dim=0)  # Average the embeddings across the sequence
    return embeddings.cpu().numpy()

# Main script
def main(fasta_file, output_dir, model_name="facebook/esm2_t36_3B_UR50D"):
    # Load sequences
    sequence_ids, sequences = load_fasta(fasta_file)

    # get fasta file name no ext
    fasta_file_name = os.path.splitext(os.path.basename(fasta_file))[0]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)


    emb_dict = {}

    # Process each sequence and save embeddings
    for seq_id, seq in tqdm(zip(sequence_ids, sequences)):
        # # check if file exists
        # if os.path.exists(os.path.join(output_dir, f"{seq_id}.npy")):
        #     continue
        # else:
        # print(f"Processing {seq_id}...")
        try:
            embedding = get_embedding(seq, model, tokenizer, device)
            emb_dict[seq_id] = embedding
        except Exception as e:
            print(f"Failed to process {seq_id}: {e}")
    
    # Save embeddings
    torch.save(emb_dict, os.path.join(output_dir, fasta_file_name + ".torch"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve ESM2 embeddings for protein sequences.")
    parser.add_argument("fasta_file", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the embeddings.")
    args = parser.parse_args()

    main(args.fasta_file, args.output_dir)
