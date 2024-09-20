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
import wandb


logger = logging.getLogger("pytorch_lightning")


def train(config: Dict):

    # Initialize Wandb Logger
    wandb_logger = WandbLogger(project="bascvi", save_dir=config["run_save_dir"])

    if "wandb_run_name" in config:
        wandb.init(name=config["wandb_run_name"], project="bascvi", dir=config["run_save_dir"])


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


    if config["datamodule"]["class_name"] == "TileDBSomaIterDataModule":
        config["datamodule"]["options"]["root_dir"] = config["run_save_dir"]
        datamodule = TileDBSomaIterDataModule(**config["datamodule"]["options"])

    elif config["datamodule"]["class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**config["datamodule"]["options"])

    elif config["datamodule"]["class_name"] == "AnnDataDataModule":
        raise NotImplementedError("Training with AnnDataDataModule is not implemented yet")
        datamodule = AnnDataDataModule(**config["datamodule"]["options"])

    datamodule.setup()

    # set the model gene list from the datamodule
    config['emb_trainer']['gene_list'] = datamodule.gene_list

    # set the number of input genes and batches in the model from the datamodule
    config['emb_trainer']['model_args']['n_input'] = datamodule.num_genes
    config['emb_trainer']['model_args']['n_batch'] = datamodule.num_batches

    # dynamically import trainer class
    module = __import__("bascvi.trainer", globals(), locals(), [config["trainer_module_name"] if "trainer_module_name" in config else "bascvi_trainer"], 0)
    EmbeddingTrainer = getattr(module, config["trainer_class_name"] if "trainer_class_name" in config else "BAScVITrainer")

    if config.get("load_from_checkpoint"):
        logger.info(f"Loading trainer from checkpoint.....")
        model = EmbeddingTrainer.load_from_checkpoint(
            config["load_from_checkpoint"], 
            )
    else:
        logger.info(f"Initializing Custom Embedding Trainer.....")
        model = EmbeddingTrainer(
            config["run_save_dir"],
            **config["emb_trainer"]
            )
    # add callbacks to pytroch lightning trainer config
    if "pl_trainer" not in config:
        config["pl_trainer"] = {}
    config["pl_trainer"]["callbacks"] = model.callbacks

    model.datamodule = datamodule

    # logger.info(f"Initializing pytorch-lightning trainer.....")
    trainer = Trainer(**config["pl_trainer"], logger=wandb_logger, accelerator="auto")
    #trainer.save_checkpoint("latest.ckpt")

    # logger.addHandler(logging.FileHandler(os.path.join(cfg["datamodule"]["root_dir"], "std.log")))


    logger.info("-----------------------Starting training-----------------------")
    trainer.fit(model, datamodule=datamodule) # Hit ctrl C to trigger predict as soon as training starts - Hack but avoids a lot of extra code
    logger.info(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Best model score: {trainer.checkpoint_callback.best_model_score}")

    trainer.save_checkpoint(os.path.join(os.path.dirname(trainer.checkpoint_callback.best_model_path), "latest.ckpt"))


    # load the best checkpoint automatically (tracked by lightning itself)
    logger.info("--------------Embedding prediction on full dataset-------------")


    # if config["datamodule_class_name"] == "TileDBSomaIterDataModule":
    #     config["datamodule"]["root_dir"] = cfg["run_save_dir"]
    #     datamodule = TileDBSomaIterDataModule(**cfg["datamodule"])
    # elif config["datamodule_class_name"] == "EmbDatamodule":
    #     datamodule = EmbDatamodule(**cfg["datamodule"])

    datamodule.pretrained_batch_size = config['emb_trainer']['model_args']['n_batch']
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
    
    # embeddings_df, fig_save_dict = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)

    save_path = os.path.join(os.path.dirname(config["run_save_dir"]), "pred_embeddings_" + os.path.splitext(os.path.basename(trainer.checkpoint_callback.best_model_path))[0] + ".tsv")
    embeddings_df.to_csv(save_path, sep="\t")
    logger.info(f"Saved predicted embeddings to: {save_path}")
