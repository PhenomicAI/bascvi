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

from bascvi.utils.utils import calc_kni_score, calc_rbni_score

warnings.filterwarnings("ignore")

import tiledbsoma as soma
import tiledb 

from dotenv import load_dotenv





# DIR = "/home/ubuntu/large-bascivi/exp_logs/scref_train"
# exps = ["baseline_no_disc"]
# files = ["/home/ubuntu/large-bascivi/exp_logs/v6/baseline_no_disc/scvi-vae-epoch=11-elbo_val=0.00.ckpt_predictions.tsv"]



def run_metrics_on_folder(root_dir: str, cell_type_col: str = "standard_true_celltype", batch_col: str = "study_name", max_prop_same_batch: float = 0.8):

    run_names = []
    pred_paths = []

    root_dir = str(Path(root_dir))

    # find all pred files
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            if "multispecies_06Nov2024_full_predictions" in file:
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


    load_dotenv("/home/ubuntu/.aws.env")

    ACCESS_KEY = os.getenv("ACCESS_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    SOMA_CORPUS_URI = "s3://pai-scrnaseq/sctx_gui/corpora/multispecies_06Nov2024/"

    soma_experiment = soma.Experiment.open(SOMA_CORPUS_URI, context=soma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx({
            "vfs.s3.aws_access_key_id": ACCESS_KEY,
            "vfs.s3.aws_secret_access_key": SECRET_KEY,
            "vfs.s3.region": "us-east-2"
        })))

    obs_df = soma_experiment.obs.read(column_names=["barcode", "standard_true_celltype"]).concat().to_pandas()
    obs_df.columns

    for emb_path in pred_paths:
        emb_df = pd.read_csv(emb_path, delimiter='\t')
        
        # drop standard_true_celltype column if it exists
        if "standard_true_celltype" in emb_df.columns:
            emb_df = emb_df.drop("standard_true_celltype", axis=1)

        emb_df = emb_df.set_index("barcode").join(obs_df.set_index("barcode"))
        print()
        print(emb_path)
        print(calc_kni_score(emb_df[cols], emb_df, batch_col=batch_col, max_prop_same_batch=max_prop_same_batch))
        print(calc_rbni_score(emb_df[cols], emb_df, batch_col=batch_col, max_prop_same_batch=max_prop_same_batch))

    # cell_types = np.asarray(df_embeddings[cell_type_col].astype('category').cat.codes,dtype=int)

    # cat = df_embeddings[batch_col].astype('category')
    # mapping = cat.cat.categories
    # study_name = cat.cat.codes
    # studies = list(range(len(mapping)))

    # results = pd.DataFrame(np.zeros((len(studies), run_names.shape[0])), index=studies, columns=run_names)

    # print("Starting Loop")


    # # KNI loop
    # for ii,fname in enumerate(tqdm(pred_paths)):

    #     df_embeddings = pd.read_csv(fname,delimiter='\t') # scVI / BAscVI / Harmony / Scanorama
    #     print(" - loaded embeddings")

    #     for i in range(dims[ii]):
    #         df_embeddings[cols[i]] = df_embeddings[cols[i]] - np.mean(df_embeddings[cols[i]])
    #         (q1,q2) = np.quantile(df_embeddings[cols[i]],[0.25,0.75])
    #         df_embeddings[cols[i]] = df_embeddings[cols[i]]/(q2-q1)
        
    #     print(" - normalized")

        
    #     cell_types = np.asarray(df_embeddings[cell_type_col].astype('category').cat.codes,dtype=int) - df_embeddings[cell_type_col].astype('category').cat.codes.min()
    #     cat = df_embeddings['study_name'].astype('category')
    #     mapping = cat.cat.categories
    #     study_name = cat.cat.codes


    #     classifier = KNeighborsClassifier(n_neighbors=50) # 25 used for csv file

    #     classifier.fit(df_embeddings[cols[:dims[ii]]], cell_types)
    #     print(" - classifier fit")

    #     vals = classifier.kneighbors(n_neighbors=50)

    #     knn_ct = cell_types[vals[1].flatten()].reshape(vals[1].shape)
    #     knn_exp = study_name.iloc[vals[1].flatten()].values.reshape(vals[1].shape)
        
    #     exp_mat = np.repeat(np.expand_dims(study_name,1),knn_exp.shape[1],axis=1)
        
    #     self_mask = knn_exp != exp_mat
    #     cutoff = np.sum(np.logical_not(self_mask),axis=1)

    #     acc = {study:0 for study in studies}
    #     batch = {study:0 for study in studies}
    #     kni = {study:0 for study in studies}

    #     mask_1 = cutoff < 40

    #     for i in tqdm(range(df_embeddings.shape[0])):
    #         if mask_1[i]:
    #             acc[study_name[i]]
    #             pred = np.argmax(np.bincount(knn_ct[i,:][self_mask[i,:]]))
    #             batch[study_name[i]] +=1
    #             if pred == cell_types[i]:
    #                 kni[study_name[i]] +=1
        
    #     print(fname)
    #     total = 0
    #     for study in studies:
    #         # print(mapping[study], '\t', kni[study])
            
    #         results[run_names[ii]].loc[study] = kni[study]
    #         total += kni[study]
    #     print("Total:  ", total, " Cell N:  ", df_embeddings.shape[0]," % Acc: " , total/df_embeddings.shape[0])
    #     print()
        
        # # Break down into accuracy vs. batch / kbet

        # print("Batch breadkdown:")
        # print()
        
        # for study in studies:
        #     print(mapping[study], '\t', batch[study])
        
        # print()
        # print("Accuracy breadkdown:")
        # print()
        
        # for study in studies:
            
        #     classifier = KNeighborsClassifier(n_neighbors=10) # 25 used for csv file
            
        #     study_mask = mapping[study] == df_embeddings['study_name']
            
        #     classifier.fit(df_embeddings[cols[:dims[ii]]].values[np.logical_not(study_mask),:], 
        #                 cell_types[np.logical_not(study_mask)])
            
        #     pred = classifier.predict(df_embeddings[cols[:dims[ii]]].values[study_mask,:])
        #     acc[study] = np.sum(pred == cell_types[study_mask])
            
        #     print(mapping[study],'\t',acc[study])
            
        # print()

    # results.to_csv(os.path.join(root_dir, "kni_results.csv"))

    # return results

    

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
        "-p",
        "--max_prop_same_batch",
        type=float,
        default=0.8
    )
    args = parser.parse_args()

    print(f"Evaluating predictions from: {args.root_dir} with batch column: {args.batch_col} and max_prop_same_batch cutoff: {args.max_prop_same_batch}")
    results = run_metrics_on_folder(args.root_dir, batch_col=args.batch_col, max_prop_same_batch=args.max_prop_same_batch)

    