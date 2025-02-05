from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os
import warnings
import umap
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import plotly.express as px

from tqdm import tqdm

from pathlib import Path

from ml_benchmarking.bascvi.utils.utils import calc_kni_score, calc_rbni_score

import tiledbsoma as soma
import tiledb 
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

import tiledbsoma as soma
import tiledb 

from dotenv import load_dotenv





# DIR = "/home/ubuntu/large-bascivi/exp_logs/scref_train"
# exps = ["baseline_no_disc"]
# files = ["/home/ubuntu/large-bascivi/exp_logs/v6/baseline_no_disc/scvi-vae-epoch=11-elbo_val=0.00.ckpt_predictions.tsv"]



def run_metrics_on_folder(root_dir: str, cell_type_col: str = "standard_true_celltype", batch_col: str = "study_name", max_prop_same_batch: float = 0.8, exclude_unknown: bool = True, restrict_species: bool = False):

    load_dotenv("/home/ubuntu/.aws.env")

    ACCESS_KEY = os.getenv("ACCESS_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    SOMA_CORPUS_URI = "s3://pai-scrnaseq/sctx_gui/corpora/multispecies_06Nov2024"

    soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
            "vfs.s3.aws_access_key_id": ACCESS_KEY,
            "vfs.s3.aws_secret_access_key": SECRET_KEY,
            "vfs.s3.region": "us-east-2"
        })))

    obs_df = soma_experiment.obs.read(column_names=["barcode", "species"]).concat().to_pandas()
    obs_df.columns


    run_names = []
    pred_paths = []

    root_dir = str(Path(root_dir))

    # find all pred files
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            if "pred_" in file:
                # get the directory after root_dir in root
                run_names.append(root.split(root_dir)[1].split("/")[1])
                pred_paths.append(os.path.join(root, file))
     
    # # Sort on name guarantee matching files to study names
    run_names = np.asarray(run_names)
    pred_paths = np.asarray(pred_paths)

    inds = np.argsort(run_names)

    run_names = run_names[inds]
    pred_paths = pred_paths[inds]

    cols = ["embedding_"+ str(i) for i in range(10)]
    dims = [10 for i in range(50)]

    # print run names and paths
    print("Run names:", run_names)
    print("Pred paths:", pred_paths)

    results_list = []

    pbar = tqdm(enumerate(pred_paths), total=len(pred_paths))
    for i, emb_path in pbar:
        if "tsv" in emb_path:
            emb_df = pd.read_csv(emb_path, delimiter='\t')
        else:
            emb_df = pd.read_csv(emb_path)

        # add species column
        if "species" not in emb_df.columns:
            emb_df = emb_df.set_index("barcode").join(obs_df.set_index("barcode"))
        
        # make metrics folder
        metrics_dir = os.path.join(root_dir, run_names[i], "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # set neurons
        neuron_list = ['Glutamatergic_neuron','Chandelier_and_Lamp5', 'Interneuron']
        emb_df[cell_type_col] = emb_df[cell_type_col].apply(lambda x: "Neuron" if x in neuron_list else x)

        if restrict_species:
            save_name = "metrics_by_batch_restrict_species.tsv"
        else:
            save_name = "metrics_by_batch.tsv"

        if restrict_species:

            run_results_by_batch = []

            # iterate over emb_df grouped by species
            for species, species_df in emb_df.groupby("species"):
            
                pbar.set_description(f"kni: {run_names[i]}")
                kni = calc_kni_score(species_df[cols], species_df, batch_col=batch_col, cell_type_col=cell_type_col, max_prop_same_batch=max_prop_same_batch, exclude_unknown=exclude_unknown)
                # pbar.set_description(f"rbni: {run_names[i]}")
                # rbni = calc_rbni_score(emb_df[cols], emb_df, batch_col=batch_col, cell_type_col=cell_type_col, max_prop_same_batch=max_prop_same_batch)

                pbar.set_description(f"saving: {run_names[i]}")

                # save confusion matrices
                confusion_matrix = kni["confusion_matrix"]
                kni_confusion_matrix = kni["kni_confusion_matrix"]

                confusion_matrix.to_csv(os.path.join(metrics_dir, f"confusion_matrix_{species}.tsv"), sep="\t")
                kni_confusion_matrix.to_csv(os.path.join(metrics_dir, f"kni_confusion_matrix_{species}.tsv"), sep="\t")

                # get results by batch
                kni_results_by_batch = kni["results_by_batch"]
                kni_results_by_batch['species'] = species

                # rbni_results_by_batch = rbni["results_by_batch"]

                # assert no identical column names
                # assert len(set(kni_results_by_batch.columns).intersection(set(rbni_results_by_batch.columns))) == 1, "Identical column names in kni and rbni results_by_batch"

                # merge kni and rbni results by batch_name
                run_results_by_batch.append(kni_results_by_batch)# pd.merge(kni_results_by_batch, rbni_results_by_batch, on="batch_name")

            results_by_batch = pd.concat(run_results_by_batch, axis=0)
            results_by_batch["model_path"] = emb_path

            # save 
            results_by_batch.to_csv(os.path.join(metrics_dir, "metrics_by_batch_restrict_species.tsv"), sep="\t")

            results_list.append(results_by_batch)
        else:
            # calculate kni and rbni scores
            pbar.set_description(f"kni: {run_names[i]}")
            kni = calc_kni_score(emb_df[cols], emb_df, batch_col=batch_col, cell_type_col=cell_type_col, max_prop_same_batch=max_prop_same_batch, exclude_unknown=exclude_unknown, n_neighbours=50) # TODO: switch to 50
            # pbar.set_description(f"rbni: {run_names[i]}")
            # rbni = calc_rbni_score(emb_df[cols], emb_df, batch_col=batch_col, cell_type_col=cell_type_col, max_prop_same_batch=max_prop_same_batch)

            pbar.set_description(f"saving: {run_names[i]}")

            # save confusion matrices
            confusion_matrix = kni["confusion_matrix"]
            kni_confusion_matrix = kni["kni_confusion_matrix"]

            confusion_matrix.to_csv(os.path.join(metrics_dir, "confusion_matrix.tsv"), sep="\t")
            kni_confusion_matrix.to_csv(os.path.join(metrics_dir, "kni_confusion_matrix.tsv"), sep="\t")

            # get results by batch
            kni_results_by_batch = kni["results_by_batch"]

            kni_results_by_batch["model_path"] = emb_path

            results_list.append(kni["results_by_batch"])
            
            print(f"Saving metrics for: {run_names[i]}")
            print_list = ['acc_knn', 'kni', 'mean_pct_same_batch_in_knn', 'pct_cells_with_diverse_knn']
            for metric in print_list:
                print(f"{metric}: {kni[metric]}")
            print("\n\n")


    results = pd.concat(results_list)

    results.to_csv(os.path.join(root_dir, save_name), sep="\t")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch_col",
        type=str,
        default="study_name"
    )
    parser.add_argument(
        "-c",
        "--cell_type_col",
        type=str,
        default="standard_true_celltype"
    )

    parser.add_argument(
        "-p",
        "--max_prop_same_batch",
        type=float,
        default=0.8
    )

    args = parser.parse_args()

    print(f"Evaluating predictions from: {args.root_dir} with batch column: {args.batch_col} and max_prop_same_batch cutoff: {args.max_prop_same_batch}")
    results = run_metrics_on_folder(args.root_dir, batch_col=args.batch_col, cell_type_col=args.cell_type_col, max_prop_same_batch=args.max_prop_same_batch, exclude_unknown=False)

    