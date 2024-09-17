from argparse import ArgumentParser
import json
import os
from pathlib import Path
import torch

from scripts.run_train import train
from scripts.run_predict import predict
from scripts.run_kni_scoring import run_kni_on_folder

def run_config(config: dict):

    # for tiledb
    torch.multiprocessing.set_start_method("fork", force=True)
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            print(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)

    if "run_save_dir" not in config:
        raise ValueError("Config must have a 'run_save_dir' key")
    
    # make sure the root dir exists
    os.makedirs(config["run_save_dir"], exist_ok=True)

    if "mode" not in config:
        raise ValueError("Config must have a 'mode' key")

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "predict":
        predict(config)
    else:
        raise ValueError(f"Invalid mode: {config['mode']}, must be one of ['train', 'predict']")

if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
    )
    args = parser.parse_args()

    # check if config path exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    with open(args.config) as json_file:
        config = json.load(json_file)

    run_config(config)
