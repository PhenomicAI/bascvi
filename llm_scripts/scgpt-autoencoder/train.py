import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from model import Autoencoder
from datamodule_emb import EmbDatamodule

import logging
import wandb
from pytorch_lightning.loggers import WandbLogger


logger = logging.getLogger("pytorch_lightning")

from soma_helpers import open_soma_experiment

def train(cfg):
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

    # Initialize data module (assuming 'dataset' is already defined)
    datamodule = EmbDatamodule(**cfg["datamodule"])


    if not cfg["prediction_checkpoint_path"]:
        # Initialize model
        model = Autoencoder()

        # Setup model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Adjust based on your validation metric
            filename="scgpt-ae-{epoch:02d}-{elbo_val:.2f}",
            save_top_k=1,
            mode='min',
        )

        # Setup early stopping
        early_stop_callback = EarlyStopping(**cfg["callbacks_args"]["early_stopping"])

        # Initialize a trainer with the checkpoint callback
        trainer = Trainer(
            max_epochs=100,
            callbacks=[checkpoint_callback, early_stop_callback],
            gpus=1,
            logger=wandb_logger
        )

        # Train the model
        trainer.fit(model, datamodule)

        checkpoint_path = trainer.checkpoint_callback.best_model_path

        logger.info(f"Best model path: {checkpoint_path}")
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    else:
        # Initialize a trainer
        trainer = Trainer(
            accelerator='gpu'
            )

        # Load from checkpoint
        model = Autoencoder.load_from_checkpoint(
            cfg["prediction_checkpoint_path"]
            )
        
        checkpoint_path = cfg["prediction_checkpoint_path"]
        
        logger.info(f"Loaded pretrained:", checkpoint_path)
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=checkpoint_path)


    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()
    emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1])[:-1]] 
    embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

    with open_soma_experiment(cfg['datamodule']['soma_experiment_uri']) as soma_experiment:
        obs_df = soma_experiment.obs.read(
                    column_names=("soma_joinid", "barcode" , "standard_true_celltype", )# "nnz", )
                ).concat().to_pandas() 
        
    embeddings_df = embeddings_df.merge(obs_df, on="soma_joinid", how="inner")



    checkpoint_name = checkpoint_path.split("/")[-1]


    # make the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    embeddings_df.to_csv(save_dir + checkpoint_name + '_ae_pred_embeddings.csv', index=False)
    logger.info(f"Saved predicted embeddings to {save_dir + checkpoint_name + '_pred_embeddings.csv'}")


    import umap

    # Extract the embedding dimensions from the merged dataframe
    embedding_dimensions = embeddings_df.loc[:, ['embedding_' + str(i) for i in range(10)]].values

    # Run UMAP on the embedding dimensions
    umap_result = umap.UMAP().fit_transform(embedding_dimensions)

    # Print the UMAP result
    # print(umap_result)

    import plotly.express as px

    # Create a DataFrame for UMAP result
    umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])

    embeddings_df.loc[embeddings_df['standard_true_celltype'] == '', 'standard_true_celltype'] = "None"

    # Add the merged_df columns to the UMAP DataFrame
    umap_df = umap_df.merge(embeddings_df, left_index=True, right_index=True)

    # Create a scatter plot using Plotly Express
    fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', color='standard_true_celltype', hover_data=['barcode'], opacity=0.5)

    # Save the plot as an image
    fig.write_image(save_dir + checkpoint_name + '_umap_plot.png')
    logger.info(f"Saved UMAP plot to {save_dir + checkpoint_name + '_umap_plot.png'}")



if __name__ == "__main__":

    # Initialize Wandb Logger
    wandb_logger = WandbLogger(project="scgpt_finetune_autoencoder", log_model=True)



    save_dir = "/home/ubuntu/scgpt-autoencoder/scmark_no_wu/finetuned/autoencoder/"

    # make the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    cfg = {
        "datamodule": {
            "emb_path": "/home/ubuntu/scgpt-autoencoder/scmark_no_wu/finetuned/scgpt_finetuned_embedding.csv",
            "num_dims": 512,
            "soma_experiment_uri": "/home/ubuntu/ml/bascvi/data/scmark/",

            "dataloader_args": {
                "batch_size": 64,
                "num_workers": 3,
                "pin_memory": True,
                "drop_last": False
            }
        },
        "prediction_checkpoint_path": "/home/ubuntu/scgpt-autoencoder/scgpt_finetune_autoencoder/r8wfquab/checkpoints/scgpt-ae-epoch=13-elbo_val=0.00.ckpt",
        "callbacks_args": {
            "model_checkpoint": {
                "monitor": "val_loss",
                "mode": "min"
            },
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 2,
                "mode": "min"
            }
        }
    }
    train(cfg)