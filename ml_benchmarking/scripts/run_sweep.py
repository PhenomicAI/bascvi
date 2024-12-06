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
            sweep_config["run_save_dir"] = os.path.join(base_root_dir, sweep_config["run_save_dir"])
        # update base config with sweep config
        for key, value in sweep_config.items():
            # check if value is a dict
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            base_config[key][k][kk] = vv
                    else:
                        base_config[key][k] = v
            else:
                base_config[key] = value

        print("Running config: ", sweep_config)

        run_config(base_config)

    run_kni_on_folder(base_root_dir)

if __name__ == '__main__':
    with open("/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/templates/macrogenes/train_scmark_test.json") as json_file:
        base_config = json.load(json_file)

    # IMPORTANT: only works with two levels of nesting
    sweep_config_list = [

        {
            "datamodule": {
                "options": {
                    "genes_to_use_path": "/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt"
                }
            },
            "emb_trainer": {
                "macrogene_method": "concat_norm",
                "training_args": {
                    "disc_loss_weight": 50000
                },
                "model_args": {
                    "n_hidden": 128,
                    "n_layers": 2,
                }
            },
            
        },

        {
            "datamodule": {
                "options": {
                    "genes_to_use_path": "/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt"
                }
            },
            "emb_trainer": {
                "macrogene_method": "concat_norm",
                "training_args": {
                    "disc_loss_weight": 50000
                },
                "model_args": {
                    "n_hidden": 256,
                    "n_layers": 2,
                }
            }
        },

                {
            "datamodule": {
                "options": {
                    "genes_to_use_path": "/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt"
                }
            },
            "emb_trainer": {
                "macrogene_method": "concat_norm",
                "training_args": {
                    "disc_loss_weight": 50000
                },
                "model_args": {
                    "n_hidden": 512,
                    "n_layers": 2,
                }
            }
        },

        # {
        #     "datamodule": {
        #         "options": {
        #             "genes_to_use_path": "/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_ortho.txt"
        #         }
        #     },
        #     "emb_trainer": {
        #         "macrogene_method": "ortholog"
        #     }
        # },

        # {
        #     "datamodule": {
        #         "options": {
        #             "genes_to_use_path": "/home/ubuntu/paper_repo/bascvi/data/human_mouse_genes_to_use_macrogenes.txt"
        #         }
        #     },
        #     "emb_trainer": {
        #         "macrogene_method": "saturn"
        #     }
        # }


    ]

    run_sweep(base_config, sweep_config_list, base_root_dir="/home/ubuntu/paper_repo/bascvi/ml_benchmarking/runs/scref_mu_macrogene_sweeps/scmark/pre_transform")