# scRNA Benchmarking with BAScVI and scMARK

This repository provides everything you need to benchmark the performance of your deep-learning architecture in aligning scRNA data. Whether you're comparing new models or testing modifications, this toolkit offers a streamlined process for evaluating embeddings and visualizing results.

## Overview

This package is built with [PyTorch](https://pytorch.org/) and leverages [PyTorch Lightning](https://www.pytorchlightning.ai/) to simplify the training and evaluation process. Familiarity with these libraries is recommended to fully understand and extend the benchmarking capabilities offered here.

The datasets used in our benchmarks are available for download at [TODO]().

## Installation

To get started, create a virtual environment and install the required dependencies:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training BAscVI

Please refer to the template config file under `ml_benchmarking/config/templates/train.json` which can be run from the working directory `bascvi/src/` using

`python -m ml_benchmarking.scripts.run_config -c ml_benchmarking/config/templates/predict_h5ad.json`

When prompted for WandDB enter (3) to proceed - unless you want to view logger results then create an account

The trainer is configured to run on a single GPU.

## Predicting with BAscVI

Please refer to the template config file under `ml_benchmarking/config/templates/predict.json` which can be run using

`python -m ml_benchmarking.scripts.run_config -c ml_benchmarking/config/templates/predict.json`

Ensure that you have downloaded our [latest checkpoint](https://huggingface.co/phenomicai/bascvi-human/resolve/main/human_bascvi_epoch_123.ckpt) and update the `pretrained_model_path` in the config file with the checkpoint location on your system.

## Key Steps

Ensure your system has enough memory for the dataloader arguments you pass. If you encounter a killed process, a good first step is to lower block_size (the number of cells loaded into memory at once) and/or decrease num_workers (the number of parallel workers).

## Models

BAscVI is based off the ScVI arcitecture, with some key adjustments detailed in our paper.

The standard ScVI models is contained in `ml_benchmarking/bascvi/model/scvi.py`. Our modified Batch-Adversarial ScVI model is in `ml_benchmarking/bascvi/model/bascvi.py`.

## Checkpoints

bascvi-human:
(TO BE UPDATED)

## Evaluation

To evaluate the performance of the embedding approach we leverage author given labels.

## How to test your own Architectures

Please refer to the template config files under `ml_benchmarking/config/templates/` for how to configure your own architecture. In the config json file, the `emb_trainer/model_args` section contains the arguments for the model architecture. The `emb_trainer/training_args` section contains the arguments for the training loop.

## License

Everything is released under an MIT license, please feel free to use it, but please cite us as we have cited others.
