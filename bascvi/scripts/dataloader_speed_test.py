from time import time

from argparse import ArgumentParser
import os
from typing import Dict
import logging
import json
import pandas as pd

from pytorch_lightning import Trainer
import torch

from bascvi.datamodule import TileDBSomaDataModule, TileDBSomaIterDataModule
from bascvi.utils.utils import umap_calc_and_save_html


with open("/home/ubuntu/large-bascivi/config/template/scmark_gf_emb.json") as json_file:
        cfg = json.load(json_file)

#cfg["datamodule"]["root_dir"] = cfg["pl_trainer"]["default_root_dir"]

soma_datamodule = EmbDatamodule(**cfg["datamodule"])

print("Set up data module....")
soma_datamodule.setup(stage="fit")
train_loader = soma_datamodule.train_dataloader()
print("begin loop")
end = time()
count = 0 
total_time = 0

print(soma_datamodule.obs_df.head())

for batch in train_loader:
    print(batch["x"])
    count += 1
    total_time += time() - end
    end = time()
    print("# s avg", count, total_time, total_time/count)

