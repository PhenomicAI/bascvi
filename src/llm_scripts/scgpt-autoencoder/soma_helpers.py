
import tiledbsoma as soma
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv("/home/ubuntu/.aws.env")

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


DEFAULT_TILEDB_CONFIGURATION = {
        "vfs.s3.aws_access_key_id": ACCESS_KEY,
        "vfs.s3.aws_secret_access_key": SECRET_KEY,
        "vfs.s3.region": "us-east-2"
    }


def _get_default_soma_context() -> soma.options.SOMATileDBContext:
    return soma.options.SOMATileDBContext().replace(tiledb_config=DEFAULT_TILEDB_CONFIGURATION)

def open_soma_experiment(tiledb_uri: str) -> soma.Experiment:
    return soma.Experiment.open(tiledb_uri, context=_get_default_soma_context())