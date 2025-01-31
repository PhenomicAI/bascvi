import numpy as np
import torch
from torch.utils.data import Dataset


class EmbTorchDataset(Dataset):
    """Custom torch dataset to get data from tiledbsoma in tensor form for pytorch modules."""
       
    def __init__(
        self,
        obs_df,
        num_samples,
        num_studies,
        num_dims,
        num_workers,
    ):     

        self.obs_df = obs_df

        self.num_dims = num_dims

        self.cell_counter = 0

        self.num_workers = num_workers

        self._len = self.obs_df.shape[0]

    def __len__(self):
        return self._len

    
    def __getitem__(self, idx):

        # get row from obs_df
        row = self.obs_df.iloc[idx]

        # get soma joinid
        soma_joinid = row["soma_joinid"]

        # get x
        x = row[["embedding_" + str(dim) for dim in range(self.num_dims)]].values
        # x = row[[str(dim) for dim in range(self.num_dims)]].values

        # make return
        ret = {
            "x": torch.from_numpy(x.astype("float32")),
            "soma_joinid": torch.tensor(soma_joinid, dtype=torch.float32),
        }

        return ret
            



