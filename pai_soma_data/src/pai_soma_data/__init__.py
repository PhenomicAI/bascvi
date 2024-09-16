from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pai_soma_data")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .pai_soma_data import SomaData

__all__ = ["SomaData"]
