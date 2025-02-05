from time import time

import json

from bascvi.datamodule.soma.datamodule import TileDBSomaIterDataModule


with open("/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/multispecies_paper/train_ms_saturn_3k.json") as json_file:
        cfg = json.load(json_file)

cfg["datamodule"]["options"]["root_dir"] = "."

soma_datamodule = TileDBSomaIterDataModule(**cfg["datamodule"]["options"])

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

