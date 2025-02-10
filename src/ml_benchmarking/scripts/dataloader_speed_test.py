from time import time
import json
from ml_benchmarking.bascvi.datamodule.soma.datamodule import TileDBSomaIterDataModule
from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment

import torch
import pytorch_lightning as pl

from tqdm import tqdm

import pandas as pd


torch.cuda.empty_cache()


with open("/home/ubuntu/paper_repo/bascvi/src/ml_benchmarking/config/multispecies_paper/predict.json") as json_file:
        cfg = json.load(json_file)

# dynamically import trainer class
module = __import__("ml_benchmarking.bascvi.trainer", globals(), locals(), [cfg["trainer_module_name"] if "trainer_module_name" in cfg else "bascvi_trainer"], 0)
EmbeddingTrainer: pl.LightningModule = getattr(module, cfg["trainer_class_name"] if "trainer_class_name" in cfg else "BAScVITrainer")


# Load the model from checkpoint
map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(cfg["pretrained_model_path"], map_location, weights_only=False)
if "gene_list" not in checkpoint['hyper_parameters']:
        raise ValueError("Pretrained model must have a 'gene_list' key in hyper_parameters")
        with open("path_to_gene_list", 'r') as f:
                pretrained_gene_list = f.read().split(",\n")
                # strip the quotes
                pretrained_gene_list = [gene.strip('"') for gene in pretrained_gene_list]
else:
        pretrained_gene_list = checkpoint['hyper_parameters']['gene_list']

n_input = checkpoint['state_dict']['vae.px_r'].shape[0]
assert n_input == len(pretrained_gene_list), f"Number of genes in the model {n_input} does not match the gene list length {len(pretrained_gene_list)}"

n_batch = checkpoint['state_dict']['vae.z_predictor.predictor.3.weight'].shape[0]

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingTrainer.load_from_checkpoint(cfg["pretrained_model_path"], map_location=lambda storage, loc: storage, root_dir=cfg["run_save_dir"], n_input=n_input, n_batch=n_batch, gene_list=pretrained_gene_list)


# Move model to device
model.to(device)

# Ensure all parameters and buffers are on the correct device
for name, param in model.named_parameters():
    if param.device != device:
        print(f"Moving {name} to {device}")
        param.data = param.data.to(device)

for name, buffer in model.named_buffers():
    if buffer.device != device:
        print(f"Moving buffer {name} to {device}")
        buffer.data = buffer.data.to(device)

# Set model to evaluation mode
model.eval()




cfg["datamodule"]["options"]["root_dir"] = "."


cfg["datamodule"]["options"]["pretrained_gene_list"] = pretrained_gene_list
cfg["datamodule"]["options"]["pretrained_batch_size"] = model.model_args.get("n_batch", None)

soma_datamodule = TileDBSomaIterDataModule(**cfg["datamodule"]["options"])

print("Set up data module....")
soma_datamodule.setup(stage="predict")
train_loader = soma_datamodule.predict_dataloader()
print("begin loop")
end = time()
count = 0 
total_time = 0

print(soma_datamodule.obs_df.head())

outputs = []

for batch in tqdm(train_loader):

        torch.cuda.empty_cache()

        # Move entire batch to the correct device
        for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

        # print(batch["x"])
        count += 1
        total_time += time() - end
        end = time()
        # print("# s avg", count, total_time, total_time/count)
        i, g = model(batch, predict_mode=True, encode=True)

        qz_m = i["qz_m"]
        z = i["z"]

        z = qz_m

        # Important: make z float64 dtype to concat properly with soma_joinid
        z = z.double()


        outputs.append(torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1).detach().cpu())
       

embeddings = torch.cat(outputs, dim=0).numpy()

emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1 )] # -1 accounts for soma_joinid
embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

with open_soma_experiment(soma_datamodule.soma_experiment_uri) as soma_experiment:
        obs_df = soma_experiment.obs.read(
                        column_names=("soma_joinid", "barcode", "standard_true_celltype", "study_name", "sample_name", "species", "scrna_protocol"),
                        ).concat().to_pandas()
                
        # merge the embeddings with the soma join id
        embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))


save_path = "pred_embeddings_" + cfg["pretrained_model_path"].split("/")[-1].split(".ckpt")[0] + ".tsv"
embeddings_df.to_csv(save_path, sep="\t")

