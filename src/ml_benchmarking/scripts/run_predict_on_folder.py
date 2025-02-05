import os
import json

from ml_benchmarking.scripts.run_config import run_config
from ml_benchmarking.scripts.run_metrics_scoring import run_metrics_on_folder


def run_predict_on_folder(base_config: dict, root_dir: str):
    # get all the directories in the root_dir with a file that starts with pred
    model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and len([f for f in os.listdir(os.path.join(root_dir, d)) if f.startswith("pred")]) > 0]

    # get subset without file that starts with config["embedding_file_name"]
    model_dirs = [d for d in model_dirs if len([f for f in os.listdir(d) if f.startswith(base_config["embedding_file_name"])]) == 0]

    print("* Model dirs without full preds: ", model_dirs)

    for model_dir in model_dirs:

        base_config["pretrained_model_path"] = False

        base_config["run_save_dir"] = model_dir

        # find .ckpt that is not latest.ckpt, may be recursive, use walk
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".ckpt") and not file.endswith("latest.ckpt"):
                    base_config["pretrained_model_path"] = os.path.join(root, file)
                    break   

        if not base_config["pretrained_model_path"]:
            print("** No model found in ", model_dir)
            continue
              
        print("** Predicting model: ", base_config["pretrained_model_path"])

        run_config(base_config)

    run_metrics_on_folder(root_dir)

if __name__ == '__main__':
    with open("/home/ubuntu/paper_repo/bascvi/ml_benchmarking/config/multispecies_paper/predict.json") as json_file:
        base_config = json.load(json_file)

    run_predict_on_folder(base_config, root_dir="/home/ubuntu/paper_repo/bascvi/ml_benchmarking/runs/ms_paper/saturn_style_3k_model_size")