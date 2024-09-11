from argparse import ArgumentParser
import os
from typing import Dict
import logging
import json
import pandas as pd

from pytorch_lightning import Trainer
import torch

from bascvi.datamodule import TileDBSomaIterDataModule, AnnDataDataModule, EmbDatamodule
from bascvi.utils.utils import umap_calc_and_save_html

from bascvi.datamodule.soma.soma_helpers import open_soma_experiment

from pytorch_lightning.loggers import WandbLogger


logger = logging.getLogger("pytorch_lightning")



def train(cfg: Dict):

    # Initialize Wandb Logger
    wandb_logger = WandbLogger(project="bascvi", log_model="all")

    # for tiledb
    torch.multiprocessing.set_start_method("fork", force=True)
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            print(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)


    if cfg["datamodule_class_name"] == "TileDBSomaIterDataModule":
        cfg["datamodule"]["root_dir"] = cfg["pl_trainer"]["default_root_dir"]
        datamodule = TileDBSomaIterDataModule(**cfg["datamodule"])

    elif cfg["datamodule_class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**cfg["datamodule"])

    elif cfg["datamodule_class_name"] == "AnnDataDataModule":
        raise NotImplementedError("Training with AnnDataDataModule is not implemented yet")
        datamodule = AnnDataDataModule(**cfg["datamodule"])

    datamodule.setup()

    # set the number of input genes and batches in the model from the datamodule
    cfg['emb_trainer']['model_args']['n_input'] = datamodule.num_genes
    cfg['emb_trainer']['model_args']['n_batch'] = datamodule.num_batches

    # dynamically import trainer class
    module = __import__("bascvi.trainer", globals(), locals(), [cfg["trainer_module_name"]], 0)
    EmbeddingTrainer = getattr(module, cfg["trainer_class_name"])

    if cfg.get("load_from_checkpoint"):
        logger.info(f"Loading trainer from checkpoint.....")
        model = EmbeddingTrainer.load_from_checkpoint(
            cfg["load_from_checkpoint"], 
            )
    else:
        logger.info(f"Initializing Custom Embedding Trainer.....")
        model = EmbeddingTrainer(
            cfg["pl_trainer"]["default_root_dir"],
            **cfg["emb_trainer"]
            )
    # add callbacks to pytroch lightning trainer config
    cfg["pl_trainer"]["callbacks"] = model.callbacks

    model.datamodule = datamodule

    # logger.info(f"Initializing pytorch-lightning trainer.....")
    trainer = Trainer(**cfg["pl_trainer"], logger=wandb_logger)  # `Trainer(accelerator='gpu', devices=1)`
    #trainer.save_checkpoint("latest.ckpt")

    # logger.addHandler(logging.FileHandler(os.path.join(cfg["datamodule"]["root_dir"], "std.log")))


    logger.info("-----------------------Starting training-----------------------")
    trainer.fit(model, datamodule=datamodule) # Hit ctrl C to trigger predict as soon as training starts - Hack but avoids a lot of extra code
    logger.info(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Best model score: {trainer.checkpoint_callback.best_model_score}")

    trainer.save_checkpoint(os.path.join(os.path.dirname(trainer.checkpoint_callback.best_model_path), "latest.ckpt"))


    # load the best checkpoint automatically (tracked by lightning itself)
    logger.info("--------------Embedding prediction on full dataset-------------")


    if cfg["datamodule_class_name"] == "TileDBSomaIterDataModule":
        cfg["datamodule"]["root_dir"] = cfg["pl_trainer"]["default_root_dir"]
        datamodule = TileDBSomaIterDataModule(**cfg["datamodule"])
    elif cfg["datamodule_class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**cfg["datamodule"])

    datamodule.pretrained_batch_size = cfg['emb_trainer']['model_args']['n_batch']
    datamodule.setup(stage="predict")

    predictions = trainer.predict(model, datamodule=datamodule)
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()

    emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
    embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

    logger.info("--------------------------Run UMAP----------------------------")
    with open_soma_experiment(datamodule.soma_experiment_uri) as soma_experiment:
        obs_df = soma_experiment.obs.read(
                            column_names=("soma_joinid", "standard_true_celltype", "sample_name", "study_name", "barcode"),
                        ).concat().to_pandas()
    obs_df
    embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))
    
    embeddings_df, fig_save_dict = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)

    save_path = os.path.join(os.path.dirname(trainer.checkpoint_callback.best_model_path), "pred_embeddings_" + os.path.splitext(os.path.basename(trainer.checkpoint_callback.best_model_path))[0] + ".tsv")
    embeddings_df.to_csv(save_path, sep="\t")
    logger.info(f"Saved predicted embeddings to: {save_path}")


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

    
    train(cfg)
