from time import time
import json
from ml_benchmarking.bascvi.datamodule.soma.datamodule import TileDBSomaIterDataModule
from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment

import torch
import pytorch_lightning as pl

from tqdm import tqdm

import pandas as pd


torch.cuda.empty_cache()


with open("/home/ubuntu/paper_repo/bascvi/src/ml_benchmarking/config/multispecies_paper/train_ms_saturn_3k.json") as json_file:
        cfg = json.load(json_file)

cfg["datamodule"]["options"]["root_dir"] = "."



soma_datamodule = TileDBSomaIterDataModule(**cfg["datamodule"]["options"])

print("Set up data module....")
soma_datamodule.setup(stage="predict")
train_loader = soma_datamodule.predict_dataloader()

print("begin loop")
end = time()
count = 0 
total_time = 0


outputs = []

for batch in tqdm(train_loader):

        count += 1
        total_time += time() - end
        end = time()
        print("# s avg", count, total_time, total_time/count)

