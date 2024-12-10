import pickle
import random
import os
import pandas as pd
import tiledbsoma as soma
import tiledb
from dotenv import load_dotenv

random.seed(42)


load_dotenv("/home/ubuntu/.aws.env")

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SOMA_CORPUS_URI = "/home/ubuntu/paper_repo/bascvi/data/corpora/multispecies_06Nov2024"

soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

obs_df = soma_experiment.obs.read().concat().to_pandas()

nuclei_ok_list = ["Brain", "Eye"]

# remove nuclei that are not ok
obs_df = obs_df[~((obs_df.cells_or_nuclei == "nuclei") & ~obs_df.tissue_collected.isin(nuclei_ok_list))]

assert ((obs_df.cells_or_nuclei == "nuclei") & obs_df.tissue_collected.isin(nuclei_ok_list)).sum() == (obs_df.cells_or_nuclei == "nuclei").sum()

print(obs_df.columns)
print(obs_df.age_stage.value_counts(dropna=False))
print(obs_df.disease_name.value_counts(dropna=False))

# remove fetal
obs_df = obs_df[obs_df.age_stage == "Adult"]

# remove disease
obs_df = obs_df[obs_df.disease_name == "Normal"]

# filter nnz
obs_df = obs_df[obs_df.nnz > 300]

print(obs_df.shape)
print(obs_df.species.value_counts(dropna=False))

print(obs_df.study_name.value_counts(dropna=False))


# sample 100k barcodes per species, balanced by species
train_barcodes = []
for species in obs_df.species.unique():
    num_studies = obs_df[obs_df.species == species].study_name.nunique()
    k = 100000//num_studies
    # sample 100k barcodes per species, balanced by study
    for study in obs_df[obs_df.species == species].study_name.unique():
        study_barcodes = obs_df[(obs_df.species == species) & (obs_df.study_name == study)].barcode.tolist()
        study_barcodes = random.sample(study_barcodes, k=k)
        train_barcodes.extend(study_barcodes)

print(len(train_barcodes))

print(obs_df[obs_df.barcode.isin(train_barcodes)].species.value_counts(dropna=False))   
print(obs_df[obs_df.barcode.isin(train_barcodes)].study_name.value_counts(dropna=False))   


with open('/home/ubuntu/paper_repo/bascvi/data/barcode_lists/multispecies_06Nov2024_train_bal_species.pkl', 'wb') as f:
    pickle.dump(train_barcodes, f)

