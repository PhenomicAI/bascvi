import os
import logging
import pandas as pd
from typing import Dict

import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from ml_benchmarking.mm_bascvi.datamodule import TileDBSomaIterDataModule, AnnDataDataModule, EmbDatamodule
from ml_benchmarking.mm_bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.utils.utils import calc_kni_score, calc_rbni_score #, umap_calc_and_save_html

from ml_benchmarking.mm_bascvi.trainer.mmbascvi_trainer import MMBAscVITrainer


logger = logging.getLogger("pytorch_lightning")


def train(config: Dict):
    
    if "wandb_project_name" not in config:
        config["wandb_project_name"] = "mm_bascvi"

    # Initialize Wandb Logger
    wandb_logger = WandbLogger(project=config["wandb_project_name"], save_dir=config["run_save_dir"])
    wandb.init(project=config["wandb_project_name"], dir=config["run_save_dir"], config=config)

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
        config["datamodule"]["options"]["root_dir"] = config["run_save_dir"]
        datamodule = EmbDatamodule(**config["datamodule"]["options"])

    elif config["datamodule"]["class_name"] == "AnnDataDataModule":
        raise NotImplementedError("Training with AnnDataDataModule is not implemented yet")
        datamodule = AnnDataDataModule(**config["datamodule"]["options"])

    datamodule.setup()

    # set the model gene list from the datamodule
    config['emb_trainer']['gene_list'] = datamodule.gene_list

    # set the number of input genes and batches in the model from the datamodule
    config['emb_trainer']['model_args']['n_input'] = datamodule.num_genes
    config['emb_trainer']['model_args']['batch_level_sizes'] = datamodule.batch_level_sizes
    config['emb_trainer']['modalities_idx_to_name_dict'] = datamodule.modalities_idx_to_name_dict


    config["emb_trainer"]["soma_experiment_uri"] = datamodule.soma_experiment_uri

    if config.get("load_from_checkpoint"):
        logger.info(f"Loading trainer from checkpoint.....")
        model = MMBAscVITrainer.load_from_checkpoint(
            config["load_from_checkpoint"], 
            )
    else:
        logger.info(f"Initializing Custom Embedding Trainer.....")
        model = MMBAscVITrainer(
            config["run_save_dir"],
            **config["emb_trainer"]
            )
    # add callbacks to pytroch lightning trainer config
    if "pl_trainer" not in config:
        config["pl_trainer"] = {}
    config["pl_trainer"]["callbacks"] = model.callbacks

    model.datamodule = datamodule

    # logger.info(f"Initializing pytorch-lightning trainer.....")
    trainer = Trainer(**config["pl_trainer"], logger=wandb_logger, accelerator="gpu", devices=1, num_sanity_val_steps=2)
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

    datamodule.pretrained_batch_size = datamodule.num_batches
    datamodule.setup(stage="predict")

    predictions = trainer.predict(model, datamodule=datamodule)
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()

    emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
    embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

    logger.info("--------------------------Run UMAP----------------------------")
    with open_soma_experiment(datamodule.soma_experiment_uri) as soma_experiment:
        obs_df = soma_experiment.obs.read(
                            column_names=("soma_joinid", "standard_true_celltype", "sample_name", "study_name", "barcode", "scrnaseq_protocol"),
                        ).concat().to_pandas()
    embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))

    obs_df = obs_df.set_index("soma_joinid").loc[embeddings_df.index]
    
    # embeddings_df, fig_save_dict = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)

    save_path = os.path.join(config["run_save_dir"], "pred_embeddings_" + os.path.splitext(os.path.basename(trainer.checkpoint_callback.best_model_path))[0] + ".tsv")
    embeddings_df.to_csv(save_path, sep="\t")
    logger.info(f"Saved predicted embeddings to: {save_path}")

    # run metrics on the embeddings, and log to wandb
    logger.info("--------------------------Run Metrics----------------------------")
    kni_score = calc_kni_score(embeddings_df[emb_columns], obs_df)
    logger.info(f"KNI Score: {kni_score}")

    rbni_score = calc_rbni_score(embeddings_df[emb_columns], obs_df)
    logger.info(f"RBNI Score: {rbni_score}")

    # plot confusion matrix
    confusion_matrix = kni_score["confusion_matrix"]
    # wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(confusion_matrix, class_names=confusion_matrix.index)})

    # drop the confusion matrix from the kni_score dict
    kni_score.pop("confusion_matrix")
    kni_score.pop("kni_confusion_matrix")
    kni_score.pop("results_by_batch")
    # kni_score.pop("non_diverse")
    # kni_score.pop("non_diverse_correctly_predicted")
    # kni_score.pop("non_diverse_incorrectly_predicted")

    rbni_score.pop("results_by_batch")


    wandb.run.summary.update(kni_score)
    wandb.run.summary.update(rbni_score)

    # end the wandb run
    wandb.finish()
