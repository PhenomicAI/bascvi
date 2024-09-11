# phenomic-ai

A CLI package which facilitates computational biologists with a single cell RNA embedding tool

## Installation

`$ pip install phenomic-ai`

## Usage

```bash
pai embed --tmp-dir [tmp_dir_path] --h5ad-path [path to h5ad file] --tissue-organ [tissue organ]

python3 -m pai embed --tmp-dir [tmp_dir_path] --h5ad-path [path to h5ad file] --tissue-organ [tissue organ]
```

Commands:

- `pai` main command
- `embed` sub-command invoking the embedding tool

Parameters:

- `--tmp-dir` (temporary direcetory) parameter is the root output directory where the downloaded zip files (zips/) and unzipped directories (results/) will be output
- `--h5ad-path` (h5ad path) parameter is the path to the single cell RNA .h5ad file intended to be uploaded and embeded
- `--tissue-organ` (tissue/organ) parameter specifies the tissue/organ associated wrt. the single cells

## Examples

```bash
pai embed --tmp-dir /tmp/pai/embed --h5ad-path ./anndata.h5ad --tissue-organ adipose

python3 -m pai embed --tmp-dir /tmp/pai/embed --h5ad-path ./anndata.h5ad --tissue-organ adipose
```

## Help

```bash
pai embed --help

python3 -m pai embed --help
```

## Support

Email: <sctx@phenomic.ai>
