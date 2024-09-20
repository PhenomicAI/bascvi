# BAscVI

## This repo accompanies the paper "Adversarial learning enables unbiased organism-wide cross-species alignment of single-cell RNA data at scale"

https://www.biorxiv.org/content/10.1101/2024.08.11.607498v2.full

This repo contains three installable modules:

pai: An API for inference of cell-type embeddings and cell-types labels from scRNA datasets - supports H5AD formats downloadable from cellxcgene

pai_soma_data: A wrapper for TileDB needed for exploring the scREF atlas we provide here on the example notebooks we provide on TileDB in the public notebooks at Phenomic/Reading from scREF and scREF-mu

ml_benchmarking: Code needed to run and evaluate ML models on the scREF and scREF-mu banchmark


## Training BAscVI

Please refer to the template config files under `ml_benchmarking/config/templates` which can be run using `python scripts/run_config.py -c PATH_TO_CONFIG`.


## License

Everything is released under an MIT license, please feel free to use it, but please cite us as we have cited others.
