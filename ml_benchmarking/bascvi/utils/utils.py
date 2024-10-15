import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import pickle as pkl

import re
import os
from typing import List

from sklearn.neighbors import KNeighborsClassifier



from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def calc_kni_score(
          embeddings_df: pd.DataFrame, 
          obs_df: pd.DataFrame, 
          cell_type_col: str = "standard_true_celltype", 
          batch_col: str = "study_name",
          n_neighbours: int = 50,
          diversity_cutoff: float = 0.1
          ) -> float:
    """Calculates KNI score for embeddings

    Parameters
    ----------
    embeddings: pd.DataFrame
        predicted embeddings with soma index as index
    obs_df: pd.DataFrame
        full_adata.obs with cell type and batch information
    cell_type_col: str
        column to use as cell type
    batch_col: str
        column to use as batch

    Returns
    -------
    kni_score: float
        KNI score
    """
    # scale embeddings using quantile normalization
    for col in embeddings_df.columns:
        embeddings_df[col] = embeddings_df[col] - embeddings_df[col].mean()
        (q1, q2) = embeddings_df[col].quantile([0.25, 0.75])
        embeddings_df[col] = embeddings_df[col] / (q2 - q1)

    # get categories
    cell_type_cat = np.asarray(obs_df[cell_type_col].astype('category').cat.codes, dtype=int) - obs_df[cell_type_col].astype('category').cat.codes.min()
    batch_cat = obs_df[batch_col].astype('category').cat.codes

    # fit classifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbours)

    classifier.fit(embeddings_df, cell_type_cat)

    # get nearest neighbors
    vals = classifier.kneighbors(n_neighbors=n_neighbours)

    # calculate KNI score
    knn_ct = cell_type_cat[vals[1].flatten()].reshape(vals[1].shape)
    knn_batch = batch_cat.iloc[vals[1].flatten()].values.reshape(vals[1].shape)

    batch_mat = np.repeat(np.expand_dims(batch_cat, 1), knn_batch.shape[1], axis=1)

    not_same_batch_mask = knn_batch != batch_mat
    num_same_batch = np.sum(np.logical_not(not_same_batch_mask), axis=1)

    acc = {b: 0 for b in batch_cat.unique()}
    batch_counts = {b: 0 for b in batch_cat.unique()}
    kni = {b: 0 for b in batch_cat.unique()}


    diverse_neighbourhood_mask = num_same_batch < diversity_cutoff * n_neighbours

    for i in range(embeddings_df.shape[0]):
        predicted_ct_by_all_neighbour = np.argmax(np.bincount(knn_ct[i,:]))
        if predicted_ct_by_all_neighbour == cell_type_cat[i]:
                acc[batch_cat.iloc[i]] +=1

        if diverse_neighbourhood_mask[i]:
            predicted_ct_by_nonbatch_neighbour = np.argmax(np.bincount(knn_ct[i,:][not_same_batch_mask[i,:]]))
            batch_counts[batch_cat.iloc[i]] +=1
            if predicted_ct_by_nonbatch_neighbour == cell_type_cat[i]:
                kni[batch_cat.iloc[i]] +=1
            
        
    kni_total = 0.0
    acc_total = 0.0
    for b in batch_cat.unique():
        kni_total += kni[b]
        acc_total += acc[b]
    

    return {'acc': acc_total / embeddings_df.shape[0], 'kni': kni_total / embeddings_df.shape[0], 'pct_same_batch_in_neighbourhood': np.mean(num_same_batch) / n_neighbours}

   
def umap_calc_and_save_html(
    embeddings: pd.DataFrame,
    emb_columns: List,
    save_dir: str,
    color_by_columns: List[str] = ["standard_true_celltype", "study_name"],
    save_model: bool = False,
    load_model: str = '',
    max_cells: int = 200000
):

    """Calculates UMAP wrt. embeddings, embeddings, and saves figures under save_dir

    Parameters
    ----------
    embeddings: pd.DataFrame
        full_adata.obs with predicted embeddings
    emb_columns: List
        embedding columns in dataframe
    save_dir
        save directory under which the html figures should be saved
    color_by_columns: List[str]
        columns to color by in the scatter plots
    save_model: bool
        whether to save the model
    load_model: str
        path to the model to load

    Returns
    -------
    embeddings: pd.DataFrame
        embeddings with columns for umap components ("umap_0", "umap_1")
    """

    assert set(emb_columns).issubset(set(embeddings.columns)), "emb_columns not in embeddings columns"
    assert set(color_by_columns).issubset(set(embeddings.columns)), "color_by_columns not in embeddings columns"

    # run UMAP
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.15, transform_seed=42)
    model = umap_model.fit(embeddings[emb_columns])


    umap_transformed = model.transform(embeddings[emb_columns])


    # Create a DataFrame for UMAP result
    umap_df = pd.DataFrame(umap_transformed, columns=['UMAP1', 'UMAP2'])


    embeddings['UMAP1'] = umap_df['UMAP1']
    embeddings['UMAP2'] = umap_df['UMAP2']
    

    size = 3 if embeddings.shape[0] > 5000 else 5

    fig_path_dict = {}
    for col in color_by_columns:
        fig = px.scatter(embeddings, x='UMAP1', y='UMAP2', color=col, width=1000, height=800, opacity=0.7, title=col)
        fig.update_traces(marker=dict(size=size))
        fig.update_layout(legend_title_text=col)
        fig.write_image(os.path.join(save_dir, f"umap_colour_by_{col}.png"))
        fig.write_html(os.path.join(save_dir, f"umap_colour_by_{col}.html"))
        fig_path_dict[col] = os.path.join(save_dir, f"umap_colour_by_{col}.png")

    return embeddings, fig_path_dict

# emb = pd.read_csv("/home/ubuntu/scmark/exp_logs/10k_adv_1/train_embeddings.tsv",sep="\t", index_col="index")
# emb_columns = ["embedding_" + str(i) for i in range(10)]
# umap_calc_and_save_html(emb,emb_columns,"/home/ubuntu/scmark/utils/")
