import os
import logging
import pandas as pd
from typing import Dict

from pytorch_lightning import Trainer
import torch
import anndata as ad

from ml_benchmarking.bascvi.datamodule import TileDBSomaIterDataModule, AnnDataDataModule, EmbDatamodule, ZarrDataModule
from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment


logger = logging.getLogger("pytorch_lightning")


def predict(config: Dict):

    logger.info("--------------Loading model from checkpoint-------------")

    if "pretrained_model_path" not in config:
        raise ValueError("Config must have a 'pretrained_model_path' key in predict mode")

    # dynamically import trainer class
    module = __import__("ml_benchmarking.bascvi.trainer", globals(), locals(), [config["trainer_module_name"] if "trainer_module_name" in config else "bascvi_trainer"], 0)
    EmbeddingTrainer = getattr(module, config["trainer_class_name"] if "trainer_class_name" in config else "BAScVITrainer")

    # Load the model from checkpoint
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["pretrained_model_path"], map_location, weights_only=False)
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

    # Get the batch_level_sizes from the checkpoint
    if "batch_level_sizes" in checkpoint['hyper_parameters']['model_args']:
        batch_level_sizes = checkpoint['hyper_parameters']['model_args']['batch_level_sizes']
    elif "n_batch" in checkpoint['hyper_parameters']['model_args']:
        batch_level_sizes = [checkpoint['hyper_parameters']['model_args']['n_batch']]
    else:
        raise ValueError("Batch level sizes or n_batch must be in the checkpoint")
    
    model = EmbeddingTrainer.load_from_checkpoint(
        config["pretrained_model_path"], 
        root_dir=config["run_save_dir"], 
        n_input=n_input, 
        batch_level_sizes=batch_level_sizes, 
        gene_list=pretrained_gene_list,
        predict_only=True
    )


    logger.info("--------------Setting up data module--------------")
    # Set the gene list in the datamodule from the saved model
    config["datamodule"]["options"]["pretrained_gene_list"] = pretrained_gene_list
    # Set the number of batches in the datamodule from the saved model
    config["datamodule"]["options"]["pretrained_batch_size"] = 0#sum(batch_level_sizes)

    if "class_name" not in config["datamodule"]:
        config["datamodule"]["class_name"] = "TileDBSomaIterDataModule"

    # Handle ZarrDataModule specially - it doesn't need pretrained_gene_list or root_dir
    if config["datamodule"]["class_name"] == "ZarrDataModule":
        # Remove parameters that ZarrDataModule doesn't accept
        options = config["datamodule"]["options"].copy()
        options.pop("pretrained_gene_list", None)
        options.pop("root_dir", None)
        options.pop("pretrained_batch_size", None)
        datamodule = ZarrDataModule(**options)
    elif config["datamodule"]["class_name"] == "TileDBSomaIterDataModule":
        config["datamodule"]["options"]["root_dir"] = config["run_save_dir"]
        datamodule = TileDBSomaIterDataModule(**config["datamodule"]["options"])

    elif config["datamodule"]["class_name"] == "EmbDatamodule":
        datamodule = EmbDatamodule(**config["datamodule"]["options"])

    elif config["datamodule"]["class_name"] == "AnnDataDataModule":
        datamodule = AnnDataDataModule(**config["datamodule"]["options"])

    else:
        raise ValueError(f"Unknown datamodule class: {config['datamodule']['class_name']}")

    datamodule.setup(stage="predict")

    model.datamodule = datamodule

    # Create a Trainer instance with minimal configuration, since we're only predicting
    trainer = Trainer(default_root_dir=config["run_save_dir"], accelerator="auto")

    logger.info("--------------Embedding prediction on full dataset-------------")
    model.eval()
    logger.info("model in eval mode")
    dataloader = datamodule.predict_dataloader()
    logger.info("predict dataloader created")
    predictions = trainer.predict(model, dataloaders=dataloader)
    logger.info("predictions made")
    embeddings = torch.cat(predictions, dim=0).detach().cpu().numpy()

    if config["datamodule"]["class_name"] == "TileDBSomaIterDataModule":
        emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1 )] # -1 accounts for soma_joinid
        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
    elif config["datamodule"]["class_name"] == "AnnDataDataModule":
        emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 2)]
        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["file_counter", "cell_counter"])
    elif config["datamodule"]["class_name"] == "ZarrDataModule":
        emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)]  # -1 accounts for cell_idx
        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["cell_idx"])
    else:
        emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns)


    # trainer.save_checkpoint("/home/ubuntu/paper_repo/bascvi/checkpoints/paper/human_bascvi_1k/human_bascvi_epoch_123.ckpt")


    # logger.info("--------------------------Run UMAP----------------------------")
    # if embeddings_df.shape[0] > 500000:
    #     logger.info("Too many embeddings to calculate UMAP, skipping....")
    # else:

    #     embeddings_df = umap_calc_and_save_html(embeddings_df, emb_columns, trainer.default_root_dir)
    
    logger.info("-----------------------Save Embeddings------------------------")
    if config["datamodule"]["class_name"] in ["TileDBSomaIterDataModule", "EmbDatamodule"]:

        with open_soma_experiment(datamodule.soma_experiment_uri) as soma_experiment:
            obs_df = soma_experiment.obs.read(
                                column_names=("soma_joinid", "barcode", "standard_true_celltype", "authors_celltype", "cell_type_pred", "cell_subtype_pred", "sample_name", "study_name", "batch_name"),
                            ).concat().to_pandas()
                        
            # merge the embeddings with the soma join id
            embeddings_df = embeddings_df.set_index("soma_joinid").join(obs_df.set_index("soma_joinid"))
    elif config["datamodule"]["class_name"] == "AnnDataDataModule":
        file_paths_df = pd.DataFrame(datamodule.file_paths, columns=["file_path"])
        file_paths_df["file_counter"] = file_paths_df.index 
        embeddings_df = embeddings_df.merge(file_paths_df, on="file_counter")
    elif config["datamodule"]["class_name"] == "ZarrDataModule":
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
        
        logger.info(f"Embeddings shape: {embeddings_df.shape}")
        logger.info(f"Available columns: {list(embeddings_df.columns)}")

    if "embedding_file_name" not in config:
        config["embedding_file_name"] = "pred_" + config["pretrained_model_path"].split("/")[-1].split(".")[0]

    save_path = os.path.join(config["run_save_dir"], config["embedding_file_name"] + ".tsv")
    embeddings_df.to_csv(save_path, sep="\t")
    logger.info(f"Saved predicted embeddings to: {save_path}")
