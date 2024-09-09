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

logger = logging.getLogger("pytorch_lightning")

def predict(cfg: Dict, checkpoint_path: str):

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


    # get default root dir from checkpoint path
    cfg["pl_trainer"]["default_root_dir"] = os.path.dirname(checkpoint_path)


    # dynamically import trainer class
    module = __import__("bascvi.trainer", globals(), locals(), [cfg["trainer_module_name"]], 0)
    EmbeddingTrainer = getattr(module, cfg["trainer_class_name"])

    # Load the model from checkpoint
    model = EmbeddingTrainer.load_from_checkpoint(checkpoint_path, root_dir=cfg["pl_trainer"]["default_root_dir"])

    logger.info("Set up data module....")
    # Set the number of batches in the datamodule from the saved model
    cfg["datamodule"]["pretrained_batch_size"] = model.model_args.get("n_batch", None)

    if cfg["datamodule_class_name"] == "TileDBSomaIterDataModule":
        cfg["datamodule"]["root_dir"] = cfg["pl_trainer"]["default_root_dir"]
        datamodule = TileDBSomaIterDataModule(**cfg["datamodule"])

    elif cfg["datamodule_class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**cfg["datamodule"])

    elif cfg["datamodule_class_name"] == "AnnDataDataModule":
        datamodule = AnnDataDataModule(**cfg["datamodule"])

    datamodule.setup(stage="predict")

    model.datamodule = datamodule

    # Create a Trainer instance with minimal configuration, since we're only predicting
    trainer = Trainer(default_root_dir=cfg["pl_trainer"]["default_root_dir"])

    logger.info("--------------Embedding prediction on full dataset-------------")
    predictions = trainer.predict(model, datamodule=datamodule)
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()
    emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1 )] # -1 accounts for soma_joinid 
    embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
    
    # logger.info("--------------------------Run UMAP----------------------------")
    # if embeddings_df.shape[0] > 500000:
    #     logger.info("Too many embeddings to calculate UMAP, skipping....")
    # else:

    #     embeddings_df = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)
    
    logger.info("-----------------------Save Embeddings------------------------")
    with open_soma_experiment(datamodule.soma_experiment_uri) as soma_experiment:
        obs_df = soma_experiment.obs.read(
                            column_names=("soma_joinid", "barcode", "standard_true_celltype", "authors_celltype", "cell_type_pred", "cell_subtype_pred", "sample_name", "study_name"),
                        ).concat().to_pandas()
        embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))

    save_path = os.path.join(os.path.dirname(checkpoint_path), "pred_embeddings_" + os.path.splitext(os.path.basename(checkpoint_path))[0] + ".tsv")
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
    parser.add_argument(
        "-l",
        "--checkpoint_path",
        help="Checkpoint file path, `./exp_logs/v6/baseline_no_disc/lightning_logs/version_7/checkpoints/scvi-vae-epoch=11-elbo_val=0.00.ckpt`",
        type=str,
        default="",
    )
    args = parser.parse_args()

    logger.info(f"Reading config file from location: {args.config}")
    with open(args.config) as json_file:
        cfg = json.load(json_file)


    predict(cfg, args.checkpoint_path)
