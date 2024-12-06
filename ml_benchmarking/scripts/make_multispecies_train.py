import pickle
import random
import os
import pandas as pd
import tiledbsoma as soma
import tiledb
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/.aws.env")

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SOMA_CORPUS_URI = "s3://pai-scrnaseq/sctx_gui/corpora/multispecies_06Nov2024"

soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

obs_df = soma_experiment.obs.read().concat().to_pandas()

print(obs_df.species.value_counts())
print(obs_df.study_name.value_counts())

# sample 100k barcodes per species, balanced by species
train_barcodes = []
for species in obs_df.species.unique():
    species_barcodes = obs_df[obs_df.species == species].barcode.tolist()
    train_barcodes.extend(random.sample(species_barcodes, 100000))

with open('/home/ubuntu/paper_repo/bascvi/data/multispecies_06Nov2024_train_bal_species.pkl', 'wb') as f:
    pickle.dump(train_barcodes, f)

