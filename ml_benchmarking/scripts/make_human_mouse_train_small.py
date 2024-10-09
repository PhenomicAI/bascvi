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
SOMA_CORPUS_URI = "s3://pai-scrnaseq/sctx_gui/corpora/combined_scref_no_ortho/"

soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    })))

with open('/home/ubuntu/saturn/eval_scref_plus_mu/data/scref_train_barcodes.pkl', 'rb') as f:
    human_train = pickle.load(f)

with open('/home/ubuntu/saturn/eval_scref_plus_mu/data/v1.3mm_train_barcodes.pkl', 'rb') as f:
    mouse_train = pickle.load(f)

obs_df = soma_experiment.obs.read().concat().to_pandas()

human_barcodes = obs_df[obs_df.barcode.isin(human_train)].sample(100000, random_state=42, replace=False).barcode.values.tolist()
mouse_barcodes = obs_df[obs_df.barcode.isin(mouse_train)].sample(100000, random_state=42, replace=False).barcode.values.tolist()

with open('human_mouse_train_small.pkl', 'wb') as f:
    pickle.dump(human_barcodes + mouse_barcodes, f)
