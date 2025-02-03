import numpy as np
import copy
import os
from typing import Dict, Optional
import pytorch_lightning as pl

from torch.utils.data import DataLoader 

import pickle
import pandas as pd

from .dataset import TileDBSomaTorchIterDataset
import tiledbsoma as soma
import tiledb
from pyarrow.lib import ArrowInvalid

import pickle
import scanpy as sc

from tqdm import tqdm
import math

from .soma_helpers import open_soma_experiment

import time


class TileDBSomaIterDataModule(pl.LightningDataModule):

    def __init__(
        self,
        soma_experiment_uri,
        root_dir,
        max_cells_per_sample = None,
        pretrained_gene_list = None,
        genes_to_use_path = None,
        genes_to_use_hvg = None,
        cells_to_use_path = None,
        barcodes_to_use_path = None,
        calc_library = False,
        block_size = 1000,
        dataloader_args: Dict = {},
        pretrained_batch_size: int = None,
        verbose: bool = False,
        exclude_ribo_mito = True,
        train_column: str = None,
        random_seed: int = 42,
        batch_keys = {"modality": "scrnaseq_protocol", "study": "study_name", "sample": "sample_idx"}
        ):
        super().__init__()

        self.soma_experiment_uri = soma_experiment_uri
        self.dataloader_args = dataloader_args
        self.root_dir = root_dir
        self.max_cells_per_sample = max_cells_per_sample
        self.pretrained_gene_list = pretrained_gene_list
        self.genes_to_use_path = genes_to_use_path
        self.genes_to_use_hvg = genes_to_use_hvg
        self.cells_to_use_path = cells_to_use_path
        self.barcodes_to_use_path = barcodes_to_use_path
        self.calc_library = calc_library
        self.block_size = block_size
        self.pretrained_batch_size = pretrained_batch_size
        self.verbose = verbose
        self.exclude_ribo_mito = exclude_ribo_mito
        self.train_column = train_column
        self.random_seed = random_seed
        self.batch_keys = batch_keys

    

    def filter_and_generate_library_calcs(self, iterative = True):

        if os.path.isdir(os.path.join(self.root_dir, "cached_calcs_and_filter")):
            print("Loading cached metadata...")

            with open(os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_soma_ids.pkl'), 'rb') as f:
                self.filter_pass_soma_ids = pickle.load(f)
            print(" - loaded cached filter pass")

            self.library_calcs = pd.read_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", 'l_means_vars.csv'))
            print(" - loaded cached library calcs")

            # check if metadata calcs are done
            if max(self.library_calcs["sample_idx"].to_list()) == max(self.samples_list):
                print("   - library calcs completed!")
                self.library_calcs.set_index("sample_idx")
                print(len(self.cells_to_use), " cells passed final filter.")
                return 
            else:
                print("   - resuming library calcs...")
                self.l_means = self.library_calcs["library_log_means"].to_list()
                self.l_vars = self.library_calcs["library_log_vars"].to_list()
                samples_run = self.library_calcs["sample_idx"].to_list()
        else:
            os.makedirs(os.path.join(self.root_dir, "cached_calcs_and_filter"), exist_ok=True)
            filter_pass_soma_ids = []
            self.l_means = []
            self.l_vars = []
            samples_run = []
        
            
        print("Generating sample metadata...")

        # TODO: ensure sample list is sorted

        if iterative:
            sample_idx = self.samples_list[len(samples_run)]
            print("starting with ", sample_idx)


            # convert genes to use to bool
            genes_to_use_bool = np.zeros(self.feature_presence_matrix.shape[1], dtype=bool)
            genes_to_use_bool[self.genes_to_use] = True


            for sample_idx_i in tqdm(range(len(samples_run), len(self.samples_list))):
                sample_idx = self.samples_list[sample_idx_i]

                feature_presence_matrix = self.feature_presence_matrix[sample_idx, :].astype(bool)

                # these are the genes that are present in this sample and in the genes_to_use list
                sample_gene_ids = np.where(feature_presence_matrix & genes_to_use_bool)[0]
                
                # read soma_ids for this sample
                soma_ids_in_sample = self.obs_df[self.obs_df["sample_idx"] == sample_idx]["soma_joinid"].values.tolist()
                # with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                #     obs_table = soma_experiment.obs.read(
                #         column_names=("soma_joinid",),
                #         value_filter=f"sample_idx == {sample_idx}",
                #     ).concat()

                #     row_coord = obs_table.column("soma_joinid").combine_chunks().to_numpy()
                
                # if no rows selected, return default
                if len(soma_ids_in_sample) < 1:
                    print("skipping calcs for ", sample_idx, ", not enough cells")
                    self.l_means.append(0)
                    self.l_vars.append(1)
                    samples_run.append(sample_idx)
                    self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                                "library_log_means": self.l_means,  
                                                "library_log_vars": self.l_vars})
                    # save                         
                    self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
                    continue

                # if no counts in selected rows, return default
                try:
                    with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                        X_curr = soma_experiment.ms["RNA"]["X"]["row_raw"].read((soma_ids_in_sample, None)).coos(shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[soma_ids_in_sample, :]
                        # filter genes to use
                        X_curr = X_curr[:, sample_gene_ids]
                except ArrowInvalid:
                    print("skipping calcs for ", sample_idx, ", not enough counts")
                    self.l_means.append(0)
                    self.l_vars.append(1)
                    samples_run.append(sample_idx)
                    self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                                "library_log_means": self.l_means,  
                                                "library_log_vars": self.l_vars})
                    # save                         
                    self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
                    continue

                # get nnz and filter
                gene_counts = X_curr.getnnz(axis=1)
                cell_mask = gene_counts > 300 
                X_curr = X_curr[cell_mask, :]
                    
                print("sample ", sample_idx, ", X shape: ", X_curr.shape)

                # calc l_mean, l_var
                self.l_means.append(log_mean(X_curr))
                self.l_vars.append(log_var(X_curr))
                samples_run.append(sample_idx)

                # apply filter mask to soma_joinids and downsample
                filter_pass_soma_ids += np.array(soma_ids_in_sample)[cell_mask].tolist()

                # save intermediate filter pass list
                with open(os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_soma_ids.pkl'), 'wb') as f:
                    pickle.dump(filter_pass_soma_ids, f)
                
                # save intermediate library calcs   
                self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                                    "library_log_means": self.l_means,
                                                    "library_log_vars": self.l_vars})                
                self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
            
        else:
            samples_run = []
            self.l_means = []
            self.l_vars = []
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                X = soma_experiment.ms["RNA"]["X"]["row_raw"].read(coords=(self.obs_df.soma_joinid.values.tolist(), )).coos(shape=(soma_experiment.obs.count, soma_experiment.ms["RNA"].var.count)).concat().to_scipy().tocsr()[self.obs_df.soma_joinid.values.tolist(), :]
                X = X[:, self.genes_to_use]
                gene_counts = X.getnnz(axis=1)
                cell_mask = gene_counts > 300
                X = X[cell_mask, :]
                filtered_obs = self.obs_df.loc[cell_mask]
                filter_pass_soma_ids = self.obs_df["soma_joinid"].to_numpy()[cell_mask].tolist()
                for sample_idx in tqdm(self.samples_list):
                    rows = (filtered_obs["sample_idx"] == sample_idx).values
                    cols = self.feature_presence_matrix[sample_idx, :].astype(bool)
                    sub_X = X[rows, :]
                    sub_X = sub_X[:, cols]
                    self.l_means.append(log_mean(sub_X))
                    self.l_vars.append(log_var(sub_X))
                    samples_run.append(sample_idx)

            self.library_calcs = pd.DataFrame({"sample_idx": samples_run, 
                                            "library_log_means": self.l_means,  
                                            "library_log_vars": self.l_vars})

        # save                         
        self.filter_pass_soma_ids = set(filter_pass_soma_ids)
        self.library_calcs.to_csv(os.path.join(self.root_dir, "cached_calcs_and_filter", "l_means_vars.csv"))
        with open(os.path.join(self.root_dir, "cached_calcs_and_filter", 'filter_pass_soma_ids.pkl'), 'wb') as f:
            pickle.dump(self.filter_pass_soma_ids, f)

        print(len(self.filter_pass_soma_ids), " cells passed final filter.")
        self.library_calcs.set_index("sample_idx")



    def setup(self, stage: Optional[str] = None):
        # read metadata
        column_names = ["soma_joinid", "barcode", self.batch_keys["study"], self.batch_keys["sample"], self.batch_keys["modality"]]

        # check if nnz in obs
        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            all_column_names = [i.name for i in soma_experiment.obs.schema] 

        if "nnz" in all_column_names:
            column_names.append("nnz")

        
        if self.train_column:
            column_names.append(self.train_column)

        for c in column_names:
            if c not in all_column_names:
                raise ValueError(f"Column {c} not found in soma_experiment")
        
        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
            self.obs_df = soma_experiment.obs.read(
                                column_names=tuple(column_names),
                            ).concat().to_pandas()
            self.var_df = soma_experiment.ms["RNA"].var.read(
                        column_names=("soma_joinid", "gene",),
                    ).concat().to_pandas()
            
            # ensure samples are unique across all studies
            temp_df = self.obs_df.drop_duplicates(subset=[self.batch_keys["study"], self.batch_keys['sample']])
            assert temp_df[self.batch_keys["sample"]].nunique() == temp_df.shape[0], "Samples are not unique across studies"

            
            # create idx columns in obs
            self.obs_df["modality_idx"] = self.obs_df["modality_idx"] if self.batch_keys["modality"] == "modality_idx" else self.obs_df[self.batch_keys["modality"]].astype('category').cat.codes
            self.obs_df["study_idx"] = self.obs_df["study_idx"] if self.batch_keys["study"] == "study_idx" else self.obs_df[self.batch_keys["study"]].astype('category').cat.codes
            self.obs_df["sample_idx"] = self.obs_df["sample_idx"] if self.batch_keys["sample"] == "sample_idx" else self.obs_df[self.batch_keys["sample"]].astype('category').cat.codes

            self.num_modalities = int(self.obs_df["modality_idx"].max() + 1)
            self.num_studies = int(self.obs_df["study_idx"].max() + 1)
            self.num_samples = int(self.obs_df["sample_idx"].max() + 1)

            self.feature_presence_matrix = soma_experiment.ms["RNA"]["feature_presence_matrix"].read().coos(shape=(self.obs_df.sample_idx.nunique(), soma_experiment.ms["RNA"].var.count)).concat().to_scipy().toarray()

        # QUESTION: should we use all batches or only those in the training sample?
        # BATCH

        # TODO: ensure studies are unique across all modalities. is this necessary?




        # GENES

        # genes to use
        if self.genes_to_use_path:
            
            # for macrogenes, need consistent gene names
            self.var_df["gene"] = self.var_df["gene"].str.lower()

            # load genes to use list
            with open(self.genes_to_use_path, "r") as f:
                self.genes_to_use = f.read().split("\n")
                # ensure all genes are lowercase
                self.genes_to_use = [g.lower() for g in self.genes_to_use]
                self.genes_to_use = list(set(self.genes_to_use).intersection(set(self.var_df["gene"].tolist())))
            self.var_df = self.var_df.set_index("gene")
            self.genes_to_use = list(set(self.var_df.loc[self.genes_to_use, :]["soma_joinid"].values.tolist()))
            self.var_df = self.var_df.reset_index()

            print("read gene list with length ", len(self.genes_to_use), len(set(self.genes_to_use)))

        elif self.genes_to_use_hvg:
            # run hvg
            print("Running HVG with n_hvg = ", self.genes_to_use_hvg)
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                with soma_experiment.axis_query(
                    measurement_name="RNA", obs_query=soma.AxisQuery(coords=(None, None))
                ) as query:
                    adata: sc.AnnData = query.to_anndata(
                        X_name="row_raw",
                        column_names={"obs": ["soma_joinid"], "var": ["soma_joinid", "gene"]},
                    )
                    sc.pp.filter_cells(adata, min_genes=300)
                    sc.pp.filter_genes(adata, min_cells=3)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(adata, n_top_genes=self.genes_to_use_hvg)
                    self.genes_to_use = adata.var[adata.var["highly_variable"]]["soma_joinid"].values.tolist()
        else:
            # # defined genes to use as overlapping
            # self.genes_to_use = np.where(np.all(self.feature_presence_matrix, axis=0))[0]     
            # if len(self.genes_to_use) < 1000:
            #     raise ValueError("Gene overlap too small")
            print("Using all genes")
            self.genes_to_use = self.var_df["soma_joinid"].values.tolist() 

        # exclude genes
        if self.exclude_ribo_mito:
            exclusion_regex = r'^(MT-|RPL|RPS|MRPL|MRPS)' # starts with one of these => mito, or ribo
            genes_to_exclude = set(self.var_df[self.var_df["gene"].str.contains(exclusion_regex)]["soma_joinid"].values)
            self.genes_to_use = list(set(self.genes_to_use).difference(genes_to_exclude))

        if self.pretrained_gene_list:
            self.pretrained_gene_list = [g.lower() for g in self.pretrained_gene_list]
            self.pretrained_gene_ids = self.var_df[self.var_df["gene"].str.lower().isin(self.pretrained_gene_list)].soma_joinid.values.tolist()

            self.genes_to_use = list(set(self.pretrained_gene_ids).intersection(set(self.genes_to_use)))
            print(f"Using pretrained gene list found {100 * len(self.genes_to_use) / len(self.pretrained_gene_list)}% overlap")

            # get gene names and sort ids by gene name
            self.var_df_sub = self.var_df[self.var_df["soma_joinid"].isin(self.genes_to_use)]
            self.var_df_sub = self.var_df_sub.sort_values("gene")

            self.soma_gene_name_list = self.var_df_sub["gene"].str.lower().values.tolist()
            self.genes_to_use = self.var_df_sub["soma_joinid"].values.tolist()

            # get indices of genes in pretrained_gene_list in order of soma_gene_name_list
            self.pretrained_gene_indices = [self.pretrained_gene_list.index(gene) for gene in self.soma_gene_name_list]

            # TODO: need to get gene_list with names (maybe, maybe not, could be in model)

            self.num_genes = len(self.pretrained_gene_list)

            print("pretrained gene indices: ", len(self.pretrained_gene_indices))
            print("pretrained gene list: ", len(self.pretrained_gene_list))
            print("max gene index: ", max(self.pretrained_gene_indices))
        else:
            # get gene names and sort ids by gene name
            self.var_df_sub = self.var_df[self.var_df["soma_joinid"].isin(self.genes_to_use)]
            self.var_df_sub = self.var_df_sub.sort_values("gene")

            self.soma_gene_name_list = self.var_df_sub["gene"].values.tolist()
            self.genes_to_use = self.var_df_sub["soma_joinid"].values.tolist()

            self.gene_list = self.soma_gene_name_list

            self.pretrained_gene_indices = None

            self.num_genes = len(self.genes_to_use)


               
        # CELLS

        self.cells_to_use = None
        # cells to use
       
        if self.barcodes_to_use_path:
            if self.cells_to_use is not None:
                raise ValueError("cannot use both cells_to_use and barcodes_to_use")
            with open(self.barcodes_to_use_path, "rb") as f:
                barcodes = pickle.load(f)
            self.cells_to_use = self.obs_df.loc[self.obs_df["barcode"].isin(barcodes)]["soma_joinid"].values.tolist()
            print("read cell list with length ", len(self.cells_to_use))
        
        if self.train_column:
            if self.cells_to_use is not None:
                raise ValueError("cannot use both train_column and barcodes_to_use/cell_to_use")
            self.cells_to_use = self.obs_df.loc[self.obs_df[self.train_column]]["soma_joinid"].values.tolist()
            print("read cell list with length ", len(self.cells_to_use))

        if self.cells_to_use is None:
            print("Using all cells found in obs")
            self.cells_to_use = self.obs_df["soma_joinid"].values.tolist()

        self.samples_list = sorted(self.obs_df["sample_idx"].unique().tolist())
        self.num_samples = len(self.samples_list) 

        # downsample
        if self.max_cells_per_sample:
            self.obs_df = self.obs_df.groupby('sample_idx').apply(lambda x: x.sample(n=self.max_cells_per_sample, random_state=self.random_seed) if x.shape[0] > self.max_cells_per_sample else x).reset_index(drop=True)
            print("\tdownsampled to ", self.obs_df.shape[0], " cells")

        # filter cells
        self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.cells_to_use)] 

        print("Before filtering: ", self.obs_df.shape[0], " cells")

        # filter nnz
        if "nnz" in self.obs_df.columns:
            pass_filter = self.obs_df["nnz"] > 300
            print("Filtering cells with nnz < 300, ", pass_filter.sum(), " cells")
            self.obs_df = self.obs_df[pass_filter] 

        # library calcs
        try:
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                self.library_calcs = soma_experiment.ms["RNA"]["sample_library_calcs"].read().concat().to_pandas()
                self.library_calcs = self.library_calcs.set_index("sample_idx")
            
        except:
            self.filter_and_generate_library_calcs(iterative = (self.obs_df.shape[0] > 500000))
            # TODO: uncomment
            # print(len(set(self.filter_pass_soma_ids)), " cells passed final filter.")
            # self.obs_df = self.obs_df[self.obs_df["soma_joinid"].isin(self.filter_pass_soma_ids)]

        # define cell_idx as range
        self.obs_df['cell_idx'] = range(self.obs_df.shape[0])

        # shuffle obs
        if stage != "predict":
            self.obs_df = self.obs_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # calculate num blocks
        MIN_VAL_BLOCKS = 2
        if self.block_size > self.obs_df.shape[0] // 5:
            self.block_size = self.obs_df.shape[0] // 5
        self.num_total_blocks = math.ceil(self.obs_df.shape[0] / self.block_size) 

        # divide blocks into train test
        if self.num_total_blocks < self.dataloader_args['num_workers']:
            self.dataloader_args['num_workers'] = self.num_total_blocks

        print(self.obs_df.study_name.value_counts(dropna=False))

        self.feature_presence_matrix = self.feature_presence_matrix[:, self.genes_to_use]
     
        self.num_cells = self.obs_df.shape[0]

        self.num_batches = self.num_modalities + self.num_studies + self.num_samples
        
        print('# Blocks: ', self.num_total_blocks)
        print('# Genes: ', self.num_genes)
        print('# Total Cells: ', self.num_cells)  

        print('# Modalities: ', self.num_modalities)
        print('# Studies: ', self.num_studies)
        print('# Samples: ', self.num_samples)

        print("Obs has ", self.obs_df.shape[0], " cells, ", self.obs_df.soma_joinid.nunique(), " unique soma_joinids")
        assert self.obs_df.soma_joinid.nunique() == self.obs_df.shape[0]
        assert self.obs_df.cell_idx.nunique() == self.obs_df.shape[0]

        self.obs_df['soma_joinid'] = self.obs_df['soma_joinid'].astype('int64')
        self.obs_df['cell_idx'] = self.obs_df['cell_idx'].astype('int64')
        
        if stage == "fit":
            
            print("Stage = Fitting")
            
            self.val_blocks = max(self.num_total_blocks//5, MIN_VAL_BLOCKS)
            self.train_blocks = self.num_total_blocks - self.val_blocks
        
            print('# Blocks: ', self.num_total_blocks, ' # for Training: ', self.train_blocks)
           
            self.train_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df[ : self.train_blocks * self.block_size],
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                block_size=self.block_size,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
                num_workers=self.dataloader_args['num_workers'],
                verbose = self.verbose
            )
            self.val_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df[self.train_blocks * self.block_size : ],
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                library_calcs=self.library_calcs,
                block_size=self.block_size,
                num_modalities=self.num_modalities,
                num_studies=self.num_studies,
                num_samples=self.num_samples,
                num_workers=self.dataloader_args['num_workers'],
                verbose = self.verbose
            )
            
            
        elif stage == "predict":
    
            print("Stage = Predicting")

            
            self.pred_dataset = TileDBSomaTorchIterDataset(
                soma_experiment_uri=self.soma_experiment_uri,
                obs_df=self.obs_df, 
                num_input=self.num_genes,
                genes_to_use=self.genes_to_use,
                feature_presence_matrix=self.feature_presence_matrix,
                block_size=self.block_size,
                num_workers=self.dataloader_args['num_workers'],
                predict_mode=True,
                pretrained_gene_indices = self.pretrained_gene_indices,
                verbose = self.verbose
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, persistent_workers=True, worker_init_fn=staggered_worker_init, **self.dataloader_args)

    def val_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.val_dataset, persistent_workers=True, worker_init_fn=staggered_worker_init, **loader_args)

    def predict_dataloader(self):
        loader_args = copy.copy(self.dataloader_args)
        return DataLoader(self.pred_dataset, **loader_args)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch
    
def staggered_worker_init(worker_id):
    """Custom worker init function to stagger the initialization."""
    time.sleep(worker_id * 0)  # Sleep for some time depending on worker_id



def log_mean(X):
    log_counts = np.log(X.sum(axis=1))
    local_mean = np.mean(log_counts).astype(np.float32)
    return local_mean

def log_var(X):
    log_counts = np.log(X.sum(axis=1))
    local_var = np.var(log_counts).astype(np.float32)
    return local_var


