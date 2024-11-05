from .run_config import run_config
from .run_kni_scoring import run_kni_on_folder
import os
import json

def run_sweep(base_config: dict, sweep_config_list: dict, base_root_dir: str = None):

    for sweep_config in sweep_config_list:
        # check if sweep config has run_save_dir
        if "run_save_dir" not in sweep_config.keys():
            sweep_config["run_save_dir"] = "_".join([str(key) + "_" + str(value) for key, value in sweep_config.items()])
        if base_root_dir:
            base_config[key] = os.path.join(base_root_dir, value)
        # update base config with sweep config
        for key, value in sweep_config.items():
            base_config[key] = value

        run_config(base_config)

    run_kni_on_folder(base_root_dir)

if __name__ == '__main__':
    with open("/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/templates/macrogenes/train_base_small.json") as json_file:
        base_config = json.load(json_file)

    sweep_config_list = [
        {
            "macrogene_embedding_model": "ESM2",
        },
        {
            "macrogene_embedding_model": "ESM1b",
        },
        {
            "macrogene_embedding_model": "protXL",
        },

    ]

    run_sweep(base_config, sweep_config_list, base_root_dir="/home/ubuntu/paper_repo/bascvi/ml_benchmarking/runs/scref_mu_macrogene_sweeps/small/protein_embedding_model")