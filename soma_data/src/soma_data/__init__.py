from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("soma_data")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Phenomic AI"
__email__ = "sctx@phenomic.ai"
__license__ = "MIT"
__description__ = "A class to handle data from a Soma experiment"

from .soma_data import SomaData

__all__ = ["SomaData"]
