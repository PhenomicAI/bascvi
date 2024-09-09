from argparse import ArgumentParser
import json
import os
from pathlib import Path
import pandas as pd

from pytorch_lightning import Trainer
import torch

from scripts.run_kni_scoring import run_kni_on_folder

from bascvi.datamodule import TileDBSomaDataModule, TileDBSomaIterDataModule, EmbDatamodule
from bascvi.utils.utils import umap_calc_and_save_html

# parser = ArgumentParser(description=__doc__)
# parser.add_argument(
#     "-c",
#     "--config_root_dir",
#     type=str,
# )
# args = parser.parse_args()

# files = os.listdir(args.config_root_dir)
# files = [f for f in files if '.json' in f]

# run_root_dir = os.path.join("exp_logs", os.path.normpath(args.config_root_dir).split(os.sep)[-1])

# print("Run root folder:", run_root_dir)
print("Config files being run:")

checkpoint_files = [
    '/home/ubuntu/bascvi/exp_logs/scref_train/gf_emb/lightning_logs/version_2/checkpoints/scvi-vae-epoch=26-elbo_val=0.00.ckpt',
    '/home/ubuntu/bascvi/exp_logs/scref_train/gf_emb_no_disc/lightning_logs/version_2/checkpoints/scvi-vae-epoch=18-elbo_val=0.00.ckpt',
    '/home/ubuntu/bascvi/exp_logs/scref_train/scgpt_emb/lightning_logs/version_5/checkpoints/scvi-vae-epoch=26-elbo_val=0.00.ckpt'
    ]

cfg_files = [
    '/home/ubuntu/bascvi/config/pal_emboj_2021/gf_emb.json',
    '/home/ubuntu/bascvi/config/pal_emboj_2021/gf_emb_no_disc.json',
    '/home/ubuntu/bascvi/config/pal_emboj_2021/scgpt_emb.json'
]

for i,f in enumerate(checkpoint_files):
    print(f"\t - {i} {f}")

print("Begin predict...")

run_root_dir = '/home/ubuntu/bascvi/exp_logs/scref_train/pal_emboj_2021_predictions'
        
for i, checkpoint_file in enumerate(checkpoint_files):

    # split path by /
    split_path = checkpoint_file.split("/")
    checkpoint_name = split_path[6]

    with open(cfg_files[i]) as json_file:
        cfg = json.load(json_file)

    cfg["datamodule"]["pretrained_batch_size"] = 7158 # TODO: this is hacky need to figure out how to handle pred mode for loading pretrained

    os.makedirs(os.path.join(run_root_dir, checkpoint_name), exist_ok=True)


    print(f"____________{i}/{len(checkpoint_files)}___{split_path[6]}____________")

    if cfg["datamodule_class_name"] == "TileDBSomaIterDataModule":
        cfg["datamodule"]["root_dir"] = os.path.join(run_root_dir, checkpoint_name)
        datamodule = TileDBSomaIterDataModule(**cfg["datamodule"])
    elif cfg["datamodule_class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**cfg["datamodule"])


    datamodule.setup(stage="predict")

    print(datamodule.num_batches)

    # dynamically import trainer class
    module = __import__("bascvi.trainer", globals(), locals(), [cfg["trainer_module_name"]], 0)
    EmbeddingTrainer = getattr(module, cfg["trainer_class_name"])
    model = EmbeddingTrainer.load_from_checkpoint(checkpoint_file, datamodule=datamodule)

    # Create a Trainer instance with minimal configuration, since we're only predicting
    trainer = Trainer()
    # trainer.default_root_dir = os.path.join(run_root_dir, checkpoint_name)

    predictions = trainer.predict(model, datamodule=datamodule)
    print("predictions done:", len(predictions))
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()
    emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1])[:-1]] 
    embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
    

    # obs_df = datamodule.soma_experiment.obs.read(
    #                     column_names=("soma_joinid", "standard_true_celltype", "authors_celltype", "cell_type_pred", "cell_subtype_pred", "sample_name", "study_name"),
    #                 ).concat().to_pandas()
    # obs_df
    # embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))
    # embeddings_df = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)



    embeddings_df.to_csv(os.path.join(run_root_dir, checkpoint_name, "pred_embeddings.tsv"), sep="\t")


# run kni
run_kni_on_folder(run_root_dir)