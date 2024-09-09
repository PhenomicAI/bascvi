import pandas as pd
from umap import UMAP
import plotly.express as px
import pickle as pkl

import re
import os
from typing import List


def umap_calc_and_save_html(
    embeddings: pd.DataFrame,
    emb_columns: List,
    save_dir: str,
    save_model: bool = False,
    load_model: str = '',
    color_by: list = ["study_name_display", "sample_name", "standard_true_celltype"],
):

    """Calculates UMAP wrt. embeddings, embeddings, and saves three figure, first coloured by cell type,
    second coloured by dataset, and third colored by sample name, under tmp_dir

    Parameters
    ----------
    embeddings: pd.DataFrame
        full_adata.obs with predicted embeddings
    emb_columns: List
        embedding columns in dataframe
    save_dir
        save directory under which the html figures should be saved

    Returns
    -------
    embeddings: pd.DataFrame
        embeddings with columns for umap components ("umap_0", "umap_1")
    """

    if not load_model:

        umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.15, transform_seed=42)
        model = umap_model.fit(embeddings[emb_columns])
    
    if save_model:
        with open(os.path.join(save_dir, f"model.pkl"), 'wb') as f_handle:
            pkl.dump(model, f_handle, protocol=pkl.HIGHEST_PROTOCOL)

    if load_model:
        with open(load_model, 'rb') as f_handle:
            model = pkl.load(f_handle)


    umap_transformed = model.transform(embeddings[emb_columns])

    _x, _y = "umap_0", "umap_1"
    embeddings[_x] = umap_transformed[:, 0]
    embeddings[_y] = umap_transformed[:, 1]

    pattern = "(external_|internal_)([a-z]+)(_\w+)"
    if "study_name" in embeddings:
        embeddings["study_name"] = embeddings["study_name"].apply(str)
        embeddings["study_name_display"] = embeddings["study_name"].apply(lambda x: re.match(pattern, x).group(2) if re.match(pattern, x) else x)
    elif "dataset_name" in embeddings:
        embeddings["study_name_display"] = embeddings["dataset_name"].apply(lambda x: re.match(pattern, x).group(2))
        
    size = 3 if embeddings.shape[0] > 5000 else 5

    for color_col in color_by:
        fig = px.scatter(embeddings, x=_x, y=_y, color=color_col, width=1000, height=800)
        fig.update_traces(marker=dict(size=size, opacity=0.5,))
        fig_path = os.path.join(save_dir, f"umap_colour_by_{color_col}.html")
        fig.write_html(fig_path)

        fig_path = os.path.join(save_dir, f"umap_colour_by_{color_col}.png")
        fig.write_image(fig_path)

    # cell_type_col = list(filter(lambda x: re.match("standard_true_celltype*", x), embeddings.columns))[0]
    # fig_1 = px.scatter(embeddings, x=_x, y=_y, color=cell_type_col, width=1000, height=800)
    # fig_2 = px.scatter(embeddings, x=_x, y=_y, color="study_name_display", width=1000, height=800)
    # fig_3 = px.scatter(embeddings, x=_x, y=_y, color="sample_name", width=1000, height=800)

    # fig_1.update_traces(marker=dict(size=size, opacity=0.5,))
    # fig_2.update_traces(marker=dict(size=size, opacity=0.5,))
    # fig_3.update_traces(marker=dict(size=size, opacity=0.5,))

    # umap_cell_type_path = os.path.join(save_dir, f"umap_colour_by_cell_type.html")
    # umap_dataset_path = os.path.join(save_dir, f"umap_colour_by_study.html")
    # umap_sample_path = os.path.join(save_dir, f"umap_colour_by_sample.html")
    # fig_1.write_html(umap_cell_type_path)
    # fig_2.write_html(umap_dataset_path)
    # fig_3.write_html(umap_sample_path)

    # umap_cell_type_path = os.path.join(save_dir, f"umap_colour_by_cell_type.png")
    # umap_dataset_path = os.path.join(save_dir, f"umap_colour_by_study.png")
    # umap_sample_path = os.path.join(save_dir, f"umap_colour_by_sample.png")
    # fig_1.write_image(umap_cell_type_path)
    # fig_2.write_image(umap_dataset_path)
    # fig_3.write_image(umap_sample_path)


    return embeddings, {"val_umap_" + color_col: os.path.join(save_dir, f"umap_colour_by_{color_col}.png") for color_col in color_by}

# emb = pd.read_csv("/home/ubuntu/scmark/exp_logs/10k_adv_1/train_embeddings.tsv",sep="\t", index_col="index")
# emb_columns = ["embedding_" + str(i) for i in range(10)]
# umap_calc_and_save_html(emb,emb_columns,"/home/ubuntu/scmark/utils/")
