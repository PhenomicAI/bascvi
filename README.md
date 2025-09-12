# BAscVI

This repository accompanies the paper:

**“Adversarial learning enables unbiased organism-wide cross-species alignment of single-cell RNA data at scale”**  
[arXiv preprint](https://arxiv.org/abs/2503.20730v1)

---

## Repository Overview

This repo contains three Python modules that can be installed separately depending on your use case:

- **pai**  
  API for inference of cell-type embeddings and cell-type labels from scRNA-seq datasets.  
  Supports `.h5ad` formats (downloadable from cellxgene).

- **pai_soma_data**  
  A wrapper around TileDB for exploring the scREF atlas, used in our public notebooks.

- **ml_benchmarking**  
  Code to run and evaluate ML models on the scREF and scREF-mu benchmarks.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PhenomicAI/bascvi.git
cd bascvi
```


### pip install pai - recomended 
```bash
pip install -U phenomic-ai  
```

### Install `pai` from local

```bash
pip install -e src/pai
```

Verify:

```python
from pai.utils.option_choices import tissue_organ_option_choices
from pai.embed import PaiEmbeddings
```

### Install `pai_soma_data`

```bash
pip install -e src/pai_soma_data
```

Verify:

```python
from pai_soma_data import pai_soma_data
```

### Install `ml_benchmarking`

```bash
pip install -e src/ml_benchmarking
```

Verify:

```python
import ml_benchmarking.bascvi as bascvi
```

You can install any combination of these modules depending on your needs.

---

## Example Usage

```python
from pai.utils.option_choices import tissue_organ_option_choices
from pai.embed import PaiEmbeddings
from pai_soma_data import pai_soma_data
import ml_benchmarking.bascvi as bascvi
```

---

## License

Everything is released under an MIT license. Please feel free to use it, but cite us as we have cited others.

---
