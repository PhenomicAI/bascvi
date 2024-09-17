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

## Running the Embedding Model

You can run a single

## Trainers

Trainers

## Models

Are based off the ScVI arcitecture.

Key adjustments are detailed in our paper.

The standard ScVI models is contained in xxx

Our modified Batch-Adversarial ScVI model is in bascvi.py

## Checkpoints

bascvi-human: 
https://huggingface.co/phenomicai/bascvi-human/resolve/main/human_bascvi_epoch_123.ckpt

## Evaluation

To evaluate the performance of the embedding approach we leverage author given labels.

## How to test your own Architectures

## License

Everything is released under an MIT license, please feel free to use it, but please cite us as we have cited others.
