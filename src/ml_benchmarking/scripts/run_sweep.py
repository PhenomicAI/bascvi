import os
import json
import copy

from ml_benchmarking.scripts.run_config import run_config
from ml_benchmarking.scripts.run_metrics_scoring import run_metrics_on_folder


def recursive_update(base_dict, mod_dict):
    """
    Recursively updates base_dict with values from mod_dict.
    """
    for key, value in mod_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value

def run_sweep(base_config: dict, sweep_config_list: list, base_root_dir: str = None):
    """
    Runs a sweep by updating a copy of base_config with each sweep_config in sweep_config_list.
    """
    for sweep_config in sweep_config_list:
        # Ensure "run_save_dir" is in the sweep config
        if "run_save_dir" not in sweep_config.keys():
            sweep_config["run_save_dir"] = "_".join([str(key) + "_" + str(value) for key, value in sweep_config.items()])
        if base_root_dir:
            sweep_config["run_save_dir"] = os.path.join(base_root_dir, sweep_config["run_save_dir"])
        
        # Make a deep copy of base_config to preserve the original
        updated_config = copy.deepcopy(base_config)
        
        # Update the copied config with sweep_config recursively
        recursive_update(updated_config, sweep_config)

        print("Running config: ", updated_config)
        run_config(updated_config)

    run_metrics_on_folder(base_root_dir)


if __name__ == '__main__':
    with open("/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/multispecies_paper/train_ms_saturn_3k.json") as json_file:
        base_config = json.load(json_file)

    # IMPORTANT: only works with two levels of nesting
    sweep_config_list = [

        {
            "emb_trainer": {
                'model_args': {"n_latent": 10, "n_layers": 1, "n_hidden": 256},       
                'training_args':{
                    'optimizer': {'lr': 5e-5, 'weight_decay': 1e-6},
                }     
            }
        },



    ]

    run_sweep(base_config, sweep_config_list, base_root_dir="/home/ubuntu/paper_repo/bascvi/ml_benchmarking/runs/ms_paper/saturn_style_3k_10k_scaling")
