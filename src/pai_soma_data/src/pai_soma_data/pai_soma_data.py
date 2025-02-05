import tiledbsoma as soma
import pandas as pd
from typing import Optional, Literal, Sequence, Union, List


class SomaData:
    """
    A class to handle data from a Soma experiment.

    Attributes:
    - corpus_uri (str): URI of the corpus
    - ctx (Optional[soma.SOMATileDBContext]): Context object
    - layer (Literal["raw", "norm"]): Whether to obtain raw or norm counts
    """

    def __init__(
        self,
        corpus_uri: str,
        ctx: Optional[soma.SOMATileDBContext] = None,
        layer: Literal["raw", "norm"] = "norm",
    ):
        if ctx is None:
            ctx = soma.SOMATileDBContext()
        self.corpus_uri = corpus_uri
        self.ctx = ctx
        self.layer = layer

        # Load observation and var data
        self.obs = self._load_obs_data()
        self.var = self._load_var_data()

    @property
    def shape(self):
        with soma.Experiment.open(self.corpus_uri, context=self.ctx) as exp:
            return (exp.obs.count, exp.ms["RNA"].var.count)

    def _load_obs_data(self):
        print("Caching obs...")
        with soma.Experiment.open(self.corpus_uri, context=self.ctx) as exp:
            obs = exp.obs.read().concat().to_pandas()
        obs_cat = obs.astype("category")
        return obs_cat

    def _load_var_data(self):
        print("Caching var...")
        with soma.Experiment.open(self.corpus_uri, context=self.ctx) as exp:
            var = exp.ms["RNA"].var.read().concat().to_pandas()

        return var

    def _validate_query(
        self,
        rows: Union[pd.Series, int, slice, Sequence[Union[pd.Series, int]]],
        cols: Union[str, List[str]],
    ):
        """
        Validates the rows and columns to ensure they meet the expected types.

        Parameters:
        - rows: Should be a Pandas Series, an int, a slice, or a sequence of these types.
        - cols: Should be a string or a list of strings (genes).

        Raises:
        - ValueError: If the validation fails.
        """
        # Validate columns
        if isinstance(cols, list):
            if not all(isinstance(col, str) for col in cols):
                raise ValueError("All columns must be strings (i.e. genes).")
        elif not isinstance(cols, (str, slice)):
            raise ValueError("Columns must be a string, a list of strings, or a slice.")

        # Validate rows
        if not isinstance(rows, (pd.Series, int, slice, list, tuple)):
            raise ValueError(
                "Rows must be a Pandas Series, an integer, a slice, or a sequence of these types."
            )

        # If rows is a list or tuple, ensure all elements are valid types
        if isinstance(rows, (list, tuple)):
            if not all(isinstance(row, int) for row in rows):
                raise ValueError(
                    "Rows must be a sequence of integers if specifying a sequence."
                )

    def __getitem__(self, key):
        """
        Extracts a subset of the data based on the key provided.

        Parameters:
        - key: A tuple containing rows and columns to extract.

        Returns:
        - sc.AnnData object containing the subset of data
        """
        rows, cols = key

        # Validate rows and columns
        self._validate_query(rows, cols)

        # Convert string queries to lists
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(rows, int):
            rows = [rows]

        # Compute value filters for obs and var
        if isinstance(rows, slice):
            obs_ids = self.obs.iloc[rows]["soma_joinid"].tolist()
        else:
            obs_ids = self.obs.loc[rows, "soma_joinid"].tolist()
        obs_value_filter = f"soma_joinid in {obs_ids}"

        if isinstance(cols, slice):
            var_ids = self.var.iloc[cols]["soma_joinid"].tolist()
        else:
            var_ids = self.var.loc[self.var["gene"].isin(cols), "soma_joinid"].tolist()
        var_value_filter = f"soma_joinid in {var_ids}"

        # Determine which X_name to use based on the number of genes (columns) requested
        if len(var_ids) >= 1000:
            x_name = f"row_{self.layer}"
        else:
            x_name = f"col_{self.layer}"

        # Retrieve Anndata
        with soma.Experiment.open(self.corpus_uri, context=self.ctx) as exp:
            with exp.axis_query(
                measurement_name="RNA",
                obs_query=soma.AxisQuery(value_filter=obs_value_filter),
                var_query=soma.AxisQuery(value_filter=var_value_filter),
            ) as query:
                adata = query.to_anndata(
                    X_name=x_name,
                    obsm_layers=["umap", "embeddings"],
                    column_names={"obs": ["soma_joinid"], "var": ["gene"]},
                )
                adata.obs = pd.merge(adata.obs, self.obs, on="soma_joinid", how="left")
                return adata
