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

warnings.filterwarnings("ignore")


# DIR = "/home/ubuntu/large-bascivi/exp_logs/scref_train"
# exps = ["baseline_no_disc"]
# files = ["/home/ubuntu/large-bascivi/exp_logs/v6/baseline_no_disc/scvi-vae-epoch=11-elbo_val=0.00.ckpt_predictions.tsv"]



def run_kni_on_folder(root_dir: str, cell_type_col: str = "standard_true_celltype"):

    run_names = []
    pred_paths = []

    root_dir = str(Path(root_dir))

    # find all pred files
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for file in files:
            if "pred_embeddings" in file:
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

    # Set up for KNI
    df_embeddings = pd.read_csv(pred_paths[0], delimiter='\t')

    cell_types = np.asarray(df_embeddings[cell_type_col].astype('category').cat.codes,dtype=int)

    cat = df_embeddings['study_name'].astype('category')
    mapping = cat.cat.categories
    study_name = cat.cat.codes
    studies = list(range(len(mapping)))

    results = pd.DataFrame(np.zeros((len(studies), run_names.shape[0])), index=studies, columns=run_names)

    print("Starting Loop")


    # KNI loop
    for ii,fname in enumerate(tqdm(pred_paths)):

        df_embeddings = pd.read_csv(fname,delimiter='\t') # scVI / BAscVI / Harmony / Scanorama
        print(" - loaded embeddings")

        for i in range(dims[ii]):
            df_embeddings[cols[i]] = df_embeddings[cols[i]] - np.mean(df_embeddings[cols[i]])
            (q1,q2) = np.quantile(df_embeddings[cols[i]],[0.25,0.75])
            df_embeddings[cols[i]] = df_embeddings[cols[i]]/(q2-q1)
        
        print(" - normalized")

        
        cell_types = np.asarray(df_embeddings[cell_type_col].astype('category').cat.codes,dtype=int) - df_embeddings[cell_type_col].astype('category').cat.codes.min()
        cat = df_embeddings['study_name'].astype('category')
        mapping = cat.cat.categories
        study_name = cat.cat.codes


        classifier = KNeighborsClassifier(n_neighbors=50) # 25 used for csv file

        classifier.fit(df_embeddings[cols[:dims[ii]]], cell_types)
        print(" - classifier fit")

        vals = classifier.kneighbors(n_neighbors=50)

        knn_ct = cell_types[vals[1].flatten()].reshape(vals[1].shape)
        knn_exp = study_name.iloc[vals[1].flatten()].values.reshape(vals[1].shape)
        
        exp_mat = np.repeat(np.expand_dims(study_name,1),knn_exp.shape[1],axis=1)
        
        self_mask = knn_exp != exp_mat
        cutoff = np.sum(np.logical_not(self_mask),axis=1)

        acc = {study:0 for study in studies}
        batch = {study:0 for study in studies}
        kni = {study:0 for study in studies}

        mask_1 = cutoff < 40

        for i in tqdm(range(df_embeddings.shape[0])):
            if mask_1[i]:
                acc[study_name[i]]
                pred = np.argmax(np.bincount(knn_ct[i,:][self_mask[i,:]]))
                batch[study_name[i]] +=1
                if pred == cell_types[i]:
                    kni[study_name[i]] +=1
        
        print(fname)
        total = 0
        for study in studies:
            print(mapping[study], '\t', kni[study])
            
            results[run_names[ii]].loc[study] = kni[study]
            total += kni[study]
        print("Total:  ", total, " Cell N:  ", df_embeddings.shape[0]," % Acc: " , total/df_embeddings.shape[0])
        print()
        
        # Break down into accuracy vs. batch / kbet

        print("Batch breadkdown:")
        print()
        
        for study in studies:
            print(mapping[study], '\t', batch[study])
        
        print()
        print("Accuracy breadkdown:")
        print()
        
        for study in studies:
            
            classifier = KNeighborsClassifier(n_neighbors=10) # 25 used for csv file
            
            study_mask = mapping[study] == df_embeddings['study_name']
            
            classifier.fit(df_embeddings[cols[:dims[ii]]].values[np.logical_not(study_mask),:], 
                        cell_types[np.logical_not(study_mask)])
            
            pred = classifier.predict(df_embeddings[cols[:dims[ii]]].values[study_mask,:])
            acc[study] = np.sum(pred == cell_types[study_mask])
            
            print(mapping[study],'\t',acc[study])
            
        print()

    results.to_csv(os.path.join(root_dir, "kni_results.csv"))

    return results

    

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
    )
    args = parser.parse_args()

    print(f"Evaluating predictions from: {args.root_dir}")

    results = run_kni_on_folder(args.root_dir)

    