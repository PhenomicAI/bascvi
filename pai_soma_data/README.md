# Pai Soma Data

## Description

This package provides a class, `SomaData`, for handling data from a Soma experiment produced by Phenomic using [TileDBSOMA](https://github.com/single-cell-data/TileDB-SOMA). It allows users to interact with cached observation data as a Pandas DataFrame and retrieve subsets of data as `AnnData` objects, with features to query raw or normalized counts.

## Features

- Load and cache observation (`obs`) and variable (`var`) data from a SOMA experiment.
- Retrieve subsets of data based on specified rows and columns.
- Automatically chooses the appropriate layer (`raw` or `norm`) based on the number of genes requested.
- Converts queried data into `AnnData` format for compatibility with downstream analysis tools.

## Installation

To use this package, you'll need Python 3.8 or later.

```
pip install pai-soma-data
```

## Usage

### Initializing the SomaData Class

To start using the `SomaData` class, initialize it with the URI of your SOMA experiment. You can choose the data layer (`raw` or `norm`) based on your needs:

```
from pai_soma_data import SomaData

# Instantiate SomaData with normalized counts
sdata = SomaData(
    corpus_uri="...",
    layer="norm"  # Change to "raw" for raw counts
)
```

### Accessing Metadata

The observation and gene metadata are stored in the .obs and .var attributes of the SomaData object, respectively. These are accessible as Pandas DataFrames:

```
# View the first few rows of observation metadata
print(sdata.obs.head())

# View the first few rows of variable (gene) metadata
print(sdata.var.head())
```

### Slicing X Data

The SomaData class allows for Pandas-style slicing of data using the format (row_filter, col_filter). The data is fetched in the layer specified during instantiation (default is "norm" for normalized data).

- Row filter: Can be a pd.Series, list of integers, or slices.
- Column filter: Can be a gene name (string) or a list of gene names.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
