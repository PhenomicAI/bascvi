import os
import logging
import pandas as pd
import warnings
from typing import Dict

import torch
import wandb
import anndata as ad
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# Suppress WandB serialization warnings for large objects
warnings.filterwarnings("ignore", message="Serializing object of type list that is")

from ml_benchmarking.bascvi.datamodule import TileDBSomaIterDataModule, AnnDataDataModule, EmbDatamodule, ZarrDataModule
from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.utils.utils import calc_kni_score, calc_rbni_score

logger = logging.getLogger("pytorch_lightning")

def get_datamodule(config):
    """Factory for datamodule based on config."""
    options = config["datamodule"]["options"]
    class_name = config["datamodule"]["class_name"]

    # Only add root_dir for datamodules that need it
    if class_name != "ZarrDataModule":
        options["root_dir"] = config["run_save_dir"]

    # Handle ZarrDataModule specially - it needs pretrained_gene_list loaded from file
    if class_name == "ZarrDataModule":
        return ZarrDataModule(**options)

    elif class_name == "TileDBSomaIterDataModule":
        return TileDBSomaIterDataModule(**options)
    elif class_name == "EmbDatamodule":
        return EmbDatamodule(**options)
    elif class_name == "AnnDataDataModule":
        raise NotImplementedError("Training with AnnDataDataModule is not implemented yet")
    else:
        raise ValueError(f"Unknown datamodule class: {class_name}")

def get_trainer_and_model(config, datamodule, wandb_logger):
    """Initialize model and trainer from config."""

    config['emb_trainer']['gene_list'] = datamodule.gene_list
    config['emb_trainer']['model_args']['n_input'] = datamodule.num_genes
    config['emb_trainer']['model_args']['batch_level_sizes'] = datamodule.batch_level_sizes

    #config["emb_trainer"]["soma_experiment_uri"] = datamodule.soma_experiment_uri
    
    module = __import__(
        "ml_benchmarking.bascvi.trainer",
        globals(), locals(),
        [config.get("trainer_module_name", "bascvi_trainer")], 0)
    EmbeddingTrainer = getattr(module, config.get("trainer_class_name", "BAScVITrainer"))

    if config.get("load_from_checkpoint"):
        logger.info(f"Loading trainer from checkpoint.....")
        model = EmbeddingTrainer.load_from_checkpoint(config["load_from_checkpoint"])
    else:
        logger.info(f"Initializing Custom Embedding Trainer.....")
        model = EmbeddingTrainer(config["run_save_dir"], **config["emb_trainer"])
    if "pl_trainer" not in config:
        config["pl_trainer"] = {}
    config["pl_trainer"]["callbacks"] = model.callbacks
    
    # Extract max_epochs from training_args if present
    if "training_args" in config["emb_trainer"] and "max_epochs" in config["emb_trainer"]["training_args"]:
        config["pl_trainer"]["max_epochs"] = config["emb_trainer"]["training_args"]["max_epochs"]
    
    # Set validation frequency to run validation every epoch
    config["pl_trainer"]["val_check_interval"] = 1.0
    
    model.datamodule = datamodule
    trainer = Trainer(
        **config["pl_trainer"],
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=2
    )

    return trainer, model

def save_embeddings_and_metrics(trainer, datamodule, config, predictions):
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()
    emb_columns = [f"embedding_{i}" for i in range(embeddings.shape[1] - 1)]
    
    # Handle different datamodule types

    if hasattr(datamodule, 'soma_experiment_uri') and datamodule.soma_experiment_uri:
        # SOMA-based datamodule

        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
        logger.info("--------------------------Run UMAP----------------------------")

        with open_soma_experiment(datamodule.soma_experiment_uri) as soma_experiment:
            obs_df = soma_experiment.obs.read(
                column_names=("soma_joinid", "standard_true_celltype", "sample_name", "study_name", "barcode"),
            ).concat().to_pandas()

        embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))
        obs_df = obs_df.set_index("soma_joinid").loc[embeddings_df.index]
        save_path = os.path.join(
            config["run_save_dir"],
            "pred_embeddings_" + os.path.splitext(os.path.basename(trainer.checkpoint_callback.best_model_path))[0] + ".tsv"
        )
        embeddings_df.to_csv(save_path, sep="\t")
        logger.info(f"Saved predicted embeddings to: {save_path}")
        logger.info("--------------------------Run Metrics----------------------------")
        kni_score = calc_kni_score(embeddings_df[emb_columns], obs_df)
        rbni_score = calc_rbni_score(embeddings_df[emb_columns], obs_df)
        logger.info(f"KNI Score: {kni_score}")
        logger.info(f"RBNI Score: {rbni_score}")
        
        for k in ["confusion_matrix", "kni_confusion_matrix", "results_by_batch"]:
            kni_score.pop(k, None)
        rbni_score.pop("results_by_batch", None)

        wandb.run.summary.update(kni_score)
        wandb.run.summary.update(rbni_score)

    else:
        # Zarr-based datamodule - map obs data from zarr files
        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["cell_idx"])
        
        # Collect obs data from all zarr files to map to predictions
        logger.info("--------------------------Collecting obs data from zarr files----------------------------")
        all_obs_data = []
        
        # Get zarr files from datamodule
        from glob import glob
        zarr_files = sorted(glob(os.path.join(datamodule.data_root_dir, '*.zarr')))
        
        for zarr_file in zarr_files:
            adata = ad.read_zarr(zarr_file)
            obs_df = adata.obs.reset_index()
            all_obs_data.append(obs_df)
        
        # Combine all obs data
        combined_obs_df = pd.concat(all_obs_data, ignore_index=True)
        
        # Map obs data to embeddings using cell_idx (which is now a hash-based unique identifier)
        embeddings_df = embeddings_df.merge(combined_obs_df, left_on='cell_idx', right_on='cell_idx', how='left')
        
        save_path = os.path.join(
            config["run_save_dir"],
            "pred_embeddings_" + os.path.splitext(os.path.basename(trainer.checkpoint_callback.best_model_path))[0] + ".tsv"
        )
        
        embeddings_df.to_csv(save_path, sep="\t")
        logger.info(f"Saved predicted embeddings with obs data to: {save_path}")
        logger.info(f"Embeddings shape: {embeddings_df.shape}")
        logger.info(f"Available columns: {list(embeddings_df.columns)}")
    
    wandb.finish()

def train(config: Dict):

    config.setdefault("wandb_project_name", "bascvi")
    wandb_logger = WandbLogger(project=config["wandb_project_name"], save_dir=config["run_save_dir"])
    wandb.init(project=config["wandb_project_name"], dir=config["run_save_dir"], config=config)
    datamodule = get_datamodule(config)
    datamodule.setup()

    trainer, model = get_trainer_and_model(config, datamodule, wandb_logger)
    
    logger.info("-----------------------Starting training-----------------------")
    trainer.fit(model, datamodule=datamodule)
    logger.info(f"Best model path: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Best model score: {trainer.checkpoint_callback.best_model_score}")
   
    trainer.save_checkpoint(os.path.join(os.path.dirname(trainer.checkpoint_callback.best_model_path), "latest.ckpt"))
    logger.info("--------------Embedding prediction on full dataset-------------")
    
    # Use num_blocks for ZarrDataModule instead of num_batches
    if hasattr(datamodule, 'num_batches'):
        datamodule.pretrained_batch_size = datamodule.num_batches
    elif hasattr(datamodule, 'num_blocks'):
        datamodule.pretrained_batch_size = datamodule.num_blocks
    else:
        # Fallback to a reasonable default
        datamodule.pretrained_batch_size = 100
    datamodule.setup(stage="predict")
    predictions = trainer.predict(model, datamodule=datamodule)
    
    save_embeddings_and_metrics(trainer, datamodule, config, predictions)
