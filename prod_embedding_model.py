from argparse import ArgumentParser
import os
from typing import Dict
import logging
import json

from pytorch_lightning import Trainer
import torch

from datamodule import ScRNASeqDataModule
from utils.utils import umap_calc_and_save_html

logger = logging.getLogger("pytorch_lightning")
logger.addHandler(logging.FileHandler("std.log"))

class TrainerTopLevel:

    def __init__(self):
        self.model = None

    def train(self,cfg: Dict):
        scrna_datamodule = ScRNASeqDataModule(**cfg["datamodule"])
        logger.info("Set up data module....")
         
        scrna_datamodule.prod=True
        # dynamically import trainer class
        module = __import__("trainer", globals(), locals(), [cfg["trainer_module_name"]], 0)
        EmdeddingTrainer = getattr(module, cfg["trainer_class_name"])

        trainer = Trainer(**cfg["pl_trainer"])  # `Trainer(accelerator='gpu', devices=1)`
        if self.model == None:

            self.model = EmdeddingTrainer.load_from_checkpoint("~/scRNA_embed/karlsson_embed/bascvi-paper/exp_logs/BaScVI_4L_Both/lightning_logs/version_0/checkpoints/scvi-vae-epoch=31-elbo_val=3585.33.ckpt") # Specify checkpoint file loc

        
        predictions = trainer.predict(self.model, datamodule=scrna_datamodule)
        embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()

        emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1])]
        scrna_datamodule.full_train_adata.obs[emb_columns] = embeddings

        logger.info("--------------------------Run UMAP----------------------------")
        embeddings_df = umap_calc_and_save_html(scrna_datamodule.full_train_adata.obs, emb_columns, trainer.default_root_dir,load_model='exp_logs/BaScVI_4L_Both/model.pkl')
        embeddings_df.to_csv(os.path.join(trainer.default_root_dir, "train_embeddings.tsv"), sep="\t")
        logger.info(f"Saved predicted embeddings in embeddings.tsv located at {trainer.default_root_dir}/train_embeddings.tsv")

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        help="Config file path, `./configs/train_scvi_cfg.json`",
        type=str,
        default="./configs/train_scvi_cfg.json",
    )
    args = parser.parse_args()
    
    logger.info(f"Reading config file from location: {args.config}")
    with open(args.config) as json_file:
        cfg = json.load(json_file)
    
    files_ = os.listdir(cfg["datamodule"]["data_dir"])
    files = []
    for f in files_:
        files.append(os.path.join(cfg["datamodule"]["data_dir"],f))

    trainer_top = TrainerTopLevel()

    root_save_dir = cfg["pl_trainer"]["default_root_dir"]
    for i in range(3):
        cfg["datamodule"]["data_dir"] = files[i]
        cfg["pl_trainer"]["default_root_dir"] = os.path.join(root_save_dir,files_[i])
        trainer_top.train(cfg)

