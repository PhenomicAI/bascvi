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

    print(umap_transformed.shape)

    # Create a DataFrame for UMAP result
    umap_df = pd.DataFrame(umap_transformed, columns=['UMAP1', 'UMAP2'])

    print(umap_df.shape)

    embeddings['UMAP1'] = umap_df['UMAP1']
    embeddings['UMAP2'] = umap_df['UMAP2']

    print(embeddings.shape)
    

    size = 3 if embeddings.shape[0] > 5000 else 5

    fig_path_dict = {}
    for col in color_by_columns:
        fig = px.scatter(embeddings, x='UMAP1', y='UMAP2', color=col, width=1000, height=800, opacity=0.7, title=col)
        fig.update_traces(marker=dict(size=size))
        fig.update_layout(legend_title_text=col)
        fig.write_image(os.path.join(save_dir, f"umap_colour_by_{col}.png"))
        fig_path_dict[col] = os.path.join(save_dir, f"umap_colour_by_{col}.png")

    return embeddings, fig_path_dict

# emb = pd.read_csv("/home/ubuntu/scmark/exp_logs/10k_adv_1/train_embeddings.tsv",sep="\t", index_col="index")
# emb_columns = ["embedding_" + str(i) for i in range(10)]
# umap_calc_and_save_html(emb,emb_columns,"/home/ubuntu/scmark/utils/")
