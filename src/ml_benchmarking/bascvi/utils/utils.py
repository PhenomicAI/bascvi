import os
from typing import List

import faiss
import pandas as pd
import numpy as np
import plotly.express as px
from umap import UMAP
from sklearn.neighbors import KNeighborsClassifier


class FaissRadiusNeighbors:
    def __init__(self, r=1.0):
        self.r = r

    def fit(self, X, y):
        if X.shape[0] < 1000:
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif X.shape[0] < 10000:
            self.index = faiss.index_factory(X.shape[1], f"IVF20,Flat")
        elif X.shape[0] < 100000:
            self.index = faiss.index_factory(X.shape[1], f"IVF200,Flat")
        else:
            self.index = faiss.index_factory(X.shape[1], f"IVF1000,Flat")

        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))
    
    def radius_neighbors(self, X):
        results = self.index.range_search(X,thresh=self.r)
        vals = []

        for ii in range(1,results[0].shape[0]):
            inds = results[2][results[0][ii-1]:results[0][ii]]
            vals.append(inds[1:])
            
        return vals

class FaissKNeighbors:
    def __init__(self, k=50):
        self.k = k

    def fit(self, X, y):
        d = X.shape[1]
        self.index = index = faiss.IndexFlatL2()
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, 512)
        self.index.nprobe = 20
        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        distances, indices = self.index.search(X.astype(np.float32), self.k)
        return distances, indices


def scale_embeddings(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """Scales embeddings using quantile normalization

    Parameters
    ----------
    embeddings_df: pd.DataFrame
        embeddings to scale

    Returns
    -------
    embeddings_df: pd.DataFrame
        scaled embeddings
    """
    for col in embeddings_df.columns:
        embeddings_df[col] = embeddings_df[col] - embeddings_df[col].mean()
        (q1, q2) = embeddings_df[col].quantile([0.25, 0.75])
        if q2 - q1 == 0:
            embeddings_df[col] = 0
        else:
            embeddings_df[col] = embeddings_df[col] / (q2 - q1)
    return embeddings_df

def calc_kni_score(
          embeddings_df: pd.DataFrame, 
          obs_df: pd.DataFrame, 
          cell_type_col: str = "standard_true_celltype", 
          batch_col: str = "study_name",
          n_neighbours: int = 50,
          max_prop_same_batch: float = 0.8,
          exclude_unknown: bool = True,
          use_faiss: bool = False
          ) -> dict:
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
    max_prop_same_batch: float
        proportion of same batch in neighbourhood to consider as diverse neighbourhood

    Returns
    -------
    kni_score: float
        KNI score
    """

    # Reset index
    embeddings_df = embeddings_df.reset_index(drop=True)
    obs_df = obs_df.reset_index(drop=True)

    # set N/A to Unknown
    # check if N/A exists
    if obs_df[cell_type_col].isna().sum() > 0:
        # add "Unknown" to categories
        obs_df[cell_type_col] = obs_df[cell_type_col].astype('str')
        obs_df[cell_type_col] = obs_df[cell_type_col].fillna("Unknown")
        obs_df[cell_type_col] = obs_df[cell_type_col].astype('category')

    if exclude_unknown:
        # Subset to not "Unknown" cell types
        unknown_list = ["Unknown", "", 'unknown', 'nan', 'NaN', 'NA', 'na', 'N/A', 'n/a']
        embeddings_df = embeddings_df[~obs_df[cell_type_col].isin(unknown_list)]
        obs_df = obs_df[~obs_df[cell_type_col].isin(unknown_list)]

        assert embeddings_df.shape[0] == obs_df.shape[0]

 
    # scale embeddings using quantile normalization
    embeddings_df = scale_embeddings(embeddings_df)

    # get categories
    obs_df[cell_type_col] = obs_df[cell_type_col].astype('category')
    obs_df[batch_col] = obs_df[batch_col].astype('category')

    cell_type_cat = np.array(obs_df[cell_type_col].cat.codes)
    assert -1 not in cell_type_cat, "N/A cell type in cell type column"

    batch_cat = obs_df[batch_col].cat.codes
    batch_name = obs_df[batch_col].cat.categories


    # fit classifier
    if use_faiss == True:
        classifier = FaissKNeighbors(k=n_neighbours)
    else:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbours)


    classifier.fit(embeddings_df, cell_type_cat)

    # get nearest neighbors
    vals = classifier.kneighbors(embeddings_df)

    # calculate KNI score
    knn_ct = cell_type_cat[vals[1].flatten()].reshape(vals[1].shape)
    knn_batch = batch_cat.iloc[vals[1].flatten()].values.reshape(vals[1].shape)

    batch_mat = np.repeat(np.expand_dims(batch_cat, 1), knn_batch.shape[1], axis=1)

    # calculate diverse neighbourhood mask
    not_same_batch_mask = knn_batch != batch_mat
    num_same_batch = np.sum(np.logical_not(not_same_batch_mask), axis=1)
    diverse_neighbourhood_mask = num_same_batch < max_prop_same_batch * n_neighbours

    # results dict
    acc = {b: 0 for b in batch_cat.unique()}
    batch_counts = {b: 0 for b in batch_cat.unique()}
    kni = {b: 0 for b in batch_cat.unique()}
    diverse_pass = {b: 0 for b in batch_cat.unique()}

    num_cell_cats = obs_df[cell_type_col].cat.categories.shape[0]
    acc_conf_mat = np.zeros((num_cell_cats, num_cell_cats))
    kni_conf_mat = np.zeros((num_cell_cats, num_cell_cats))

    not_diverse_df = []

    for i in range(embeddings_df.shape[0]):

        # add cell to batch count
        batch_counts[batch_cat.iloc[i]] +=1

        # predict cell type using all neighbours
        predicted_ct_by_all_neighbour = np.argmax(np.bincount(knn_ct[i,:]))
        acc_conf_mat[cell_type_cat[i], predicted_ct_by_all_neighbour] += 1
        
        # if cell type is correctly predicted
        if predicted_ct_by_all_neighbour == cell_type_cat[i]:
            # add to accuracy
            acc[batch_cat.iloc[i]] +=1

        # if cell is in a diverse neighbourhood
        if diverse_neighbourhood_mask[i]:

            # add to diverse neighbourhood count
            diverse_pass[batch_cat.iloc[i]] += 1

            # if cell type is correctly predicted by non-batch neighbours
            predicted_ct_by_nonbatch_neighbour = np.argmax(np.bincount(knn_ct[i,:][not_same_batch_mask[i,:]]))
            kni_conf_mat[cell_type_cat[i], predicted_ct_by_nonbatch_neighbour] += 1

            if predicted_ct_by_nonbatch_neighbour == cell_type_cat[i]:
                # add to kni
                kni[batch_cat.iloc[i]] +=1

        # if cell is not in a diverse neighbourhood
        else:
            # add to not diverse df
            not_diverse_df.append({'batch_name': batch_cat.iloc[i], 'cell_type': cell_type_cat[i], 'predicted_cell_type': predicted_ct_by_all_neighbour})



    # make df from dicts
    results_df = pd.DataFrame([acc, kni, batch_counts, diverse_pass], index=["acc_count_knn", "kni_count", "batch_count_knn", "diverse_pass_count_knn"]).T

    # calculate total scores 
    kni_total = results_df["kni_count"].sum() / results_df["batch_count_knn"].sum()
    acc_total = results_df["acc_count_knn"].sum() / results_df["batch_count_knn"].sum()
    diverse_pass_total = results_df["diverse_pass_count_knn"].sum() / results_df["batch_count_knn"].sum()

    # format confusion matrix
    labels = obs_df[cell_type_col].cat.categories
    acc_conf_mat = pd.DataFrame(acc_conf_mat, index=labels, columns=labels)
    kni_conf_mat = pd.DataFrame(kni_conf_mat, index=labels, columns=labels)

    # add batch name to results, ensure same order as codes
    results_df["batch_name"] = obs_df[batch_col].cat.categories[results_df.index]

    # add sub scores
    results_df["kni_batch"] = results_df["kni_count"] / results_df["batch_count_knn"]
    results_df["acc_knn"] = results_df["acc_count_knn"] / results_df["batch_count_knn"]
    results_df["diverse_pass_knn"] = results_df["diverse_pass_count_knn"] / results_df["batch_count_knn"]

    # summarize non diverse cells
    not_diverse_df = pd.DataFrame(not_diverse_df)
    if not_diverse_df.shape[0] > 0:
        not_diverse_df['cell_type'] = not_diverse_df['cell_type']
        not_diverse_df['predicted_cell_type'] = not_diverse_df['predicted_cell_type']
        not_diverse_df['batch_name'] = obs_df[batch_col].cat.categories[not_diverse_df['batch_name']]
        not_diverse_df['cell_type'] = obs_df[cell_type_col].cat.categories[not_diverse_df['cell_type']]
        not_diverse_df['predicted_cell_type'] = obs_df[cell_type_col].cat.categories[not_diverse_df['predicted_cell_type']]
        # split by prediction
        non_diverse_correctly_predicted = not_diverse_df.loc[not_diverse_df['cell_type'] == not_diverse_df['predicted_cell_type']].copy()
        non_diverse_incorrectly_predicted = not_diverse_df.loc[not_diverse_df['cell_type'] != not_diverse_df['predicted_cell_type']].copy()
        # drop prediction 
        non_diverse_correctly_predicted.drop('predicted_cell_type', axis=1, inplace=True)
        non_diverse_incorrectly_predicted.drop('predicted_cell_type', axis=1, inplace=True)
        # group by batch and cell type and count
        non_diverse_correctly_predicted = non_diverse_correctly_predicted.groupby(['batch_name', 'cell_type']).size().reset_index(name='counts')
        non_diverse_incorrectly_predicted = non_diverse_incorrectly_predicted.groupby(['batch_name', 'cell_type']).size().reset_index(name='counts')


    return {
        'acc_knn': acc_total,
        'kni': kni_total,
        'mean_pct_same_batch_in_knn': np.mean(num_same_batch) / n_neighbours, 
        'pct_cells_with_diverse_knn': diverse_pass_total, 
        'confusion_matrix': acc_conf_mat,
        'kni_confusion_matrix': kni_conf_mat,
        'results_by_batch': results_df,
        }


def calc_rbni_score(
    embeddings_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    cell_type_col: str = "standard_true_celltype",
    batch_col: str = "study_name",
    radius: float = 1.0,
    max_prop_same_batch: float = 0.8,
    exclude_unknown: bool = True
    ):
    # Reset index
    embeddings_df = embeddings_df.reset_index(drop=True)
    obs_df = obs_df.reset_index(drop=True)

    if exclude_unknown:
        # Subset to not "Unknown" cell types
        embeddings_df = embeddings_df[obs_df[cell_type_col] != "Unknown"]
        obs_df = obs_df[obs_df[cell_type_col] != "Unknown"]

    # scale embeddings using quantile normalization
    embeddings_df = scale_embeddings(embeddings_df)

    # get categories
    cell_type_cat = np.asarray(obs_df[cell_type_col].astype('category').cat.codes, dtype=int) - obs_df[cell_type_col].astype('category').cat.codes.min()
    obs_df[batch_col] = obs_df[batch_col].astype('category')
    batch_cat = obs_df[batch_col].cat.codes
    batch_name = obs_df[batch_col].cat.categories

    # fit classifier
    classifier = FaissRadiusNeighbors(r=radius)
    classifier.fit(embeddings_df.values, cell_type_cat)

    # get radius neighbors
    vals = classifier.radius_neighbors(embeddings_df.values)

    # calculate rbni score
    acc = {b: 0 for b in batch_cat.unique()}
    batch_counts = {b: 0 for b in batch_cat.unique()}
    rbni = {b: 0 for b in batch_cat.unique()}
    diverse_pass = {b: 0 for b in batch_cat.unique()}
    
    global_prop_same_batch = 0.0

    for i in range(embeddings_df.shape[0]):
        neighbour_inds = vals[i]
        if len(neighbour_inds) == 0:
            continue
        # add cell to batch count
        batch_counts[batch_cat.iloc[i]] +=1

        # predict cell type using all neighbours
        predicted_ct_by_all_neighbour = np.argmax(np.bincount(cell_type_cat[neighbour_inds]))

        # if cell type is correctly predicted
        if predicted_ct_by_all_neighbour == cell_type_cat[i]:
                # add to accuracy
                acc[batch_cat.iloc[i]] += 1

        different_batch_mask = batch_cat.iloc[neighbour_inds] != batch_cat.iloc[i]
        prop_same_batch = (len(neighbour_inds) - np.sum(different_batch_mask)) / len(neighbour_inds)
        # if cell is in a diverse neighbourhood
        if prop_same_batch < max_prop_same_batch:
            # add to diverse neighbourhood count
            diverse_pass[batch_cat.iloc[i]] += 1
            # if cell type is correctly predicted by non-batch neighbours
            predicted_ct_by_nonbatch_neighbour = np.argmax(np.bincount(cell_type_cat[neighbour_inds[different_batch_mask]]))
            if predicted_ct_by_nonbatch_neighbour == cell_type_cat[i]:
                # add to rbni
                rbni[batch_cat.iloc[i]] +=1       

        global_prop_same_batch += prop_same_batch 
        
    # make df from dicts
    results_df = pd.DataFrame([acc, rbni, batch_counts, diverse_pass], index=["acc_count_radius", "rbni_count", "batch_count_radius", "diverse_pass_count_radius"]).T

    # add batch name to results, ensure same order as codes
    results_df["batch_name"] = obs_df[batch_col].cat.categories[results_df.index]

    # calculate total scores
    rbni_total = results_df["rbni_count"].sum() / (results_df["batch_count_radius"].sum() + 1e-6)
    acc_total = results_df["acc_count_radius"].sum() / (results_df["batch_count_radius"].sum() + 1e-6)
    diverse_pass = results_df["diverse_pass_count_radius"].sum() / (results_df["batch_count_radius"].sum() + 1e-6)
    global_prop_same_batch = global_prop_same_batch / embeddings_df.shape[0]

    # add sub scores
    results_df["rbni_batch"] = results_df["rbni_count"] / results_df["batch_count_radius"]
    results_df["acc_radius"] = results_df["acc_count_radius"] / results_df["batch_count_radius"]
    results_df["diverse_pass_radius"] = results_df["diverse_pass_count_radius"] / results_df["batch_count_radius"]

    return {
        'rbni': rbni_total,
        'acc_radius': acc_total,
        'mean_pct_same_batch_in_radius': global_prop_same_batch,
        'pct_cells_with_diverse_radius': diverse_pass,
        'results_by_batch': results_df
    }

def umap_calc_and_save_html(
    embeddings: pd.DataFrame,
    emb_columns: List,
    save_dir: str,
    color_by_columns: List[str] = ["standard_true_celltype", "study_name", "scrnaseq_protocol"],
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
        fig = px.scatter(embeddings, x='UMAP1', y='UMAP2', color=col, width=1000, height=800, opacity=0.2, title=col)
        fig.update_traces(marker=dict(size=size))
        fig.update_layout(legend_title_text=col)
        fig.write_image(os.path.join(save_dir, f"umap_colour_by_{col}.png"))
        fig.write_html(os.path.join(save_dir, f"umap_colour_by_{col}.html"))
        fig_path_dict[col] = os.path.join(save_dir, f"umap_colour_by_{col}.png")
        # make legend dots bigger
        fig.update_layout(legend=dict(itemsizing='constant'))

    return embeddings, fig_path_dict

# emb = pd.read_csv("/home/ubuntu/scmark/exp_logs/10k_adv_1/train_embeddings.tsv",sep="\t", index_col="index")
# emb_columns = ["embedding_" + str(i) for i in range(10)]
# umap_calc_and_save_html(emb,emb_columns,"/home/ubuntu/scmark/utils/")
