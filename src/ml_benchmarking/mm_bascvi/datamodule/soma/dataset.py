import math
from typing import Dict, List
import queue
import threading
import time
import os

import torch
import numpy as np
import tiledbsoma as soma
from torch.utils.data import IterableDataset

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment


class TileDBSomaTorchIterDataset(IterableDataset):
    """Custom torch dataset to get data from tiledbsoma in tensor form for pytorch modules with prefetching."""
       
    def __init__(
        self,
        soma_experiment_uri,
        obs_df,
        num_input,
        genes_to_use,
        feature_presence_matrix,
        block_size,
        num_workers,
        max_queue_size=300000,  # Maximum size of the shared queue
        num_modalities=None,
        num_studies=None,
        num_samples=None,
        library_calcs=None,
        verbose=False,
        predict_mode=False,
        pretrained_gene_indices=None,
        shuffle=False
    ):     
        self.soma_experiment_uri = soma_experiment_uri
        self.obs_df = obs_df
        self.genes_to_use = genes_to_use
        self.X_array_name = "row_raw"
        self.feature_presence_matrix = feature_presence_matrix
        self.predict_mode = predict_mode
        self.num_input = num_input
        self.num_modalities = num_modalities
        self.num_studies = num_studies
        self.num_samples = num_samples
        
        if self.predict_mode:
            assert self.num_modalities is None
            assert self.num_studies is None
            assert self.num_samples is None

        self.block_size = block_size
        self.num_blocks = math.ceil(self.obs_df.shape[0] / self.block_size) 
        self.library_calcs = library_calcs
        self.num_workers = num_workers
        self._len = self.obs_df.shape[0]
        self.verbose = verbose
        self.pretrained_gene_indices = pretrained_gene_indices

        # Prefetching parameters
        self.max_queue_size = max_queue_size
        
        # These will be initialized in __iter__ to avoid pickling issues
        self.data_queue = None
        self.db_lock = None
        self.worker_lock = None
        self.active_workers = 0
        self.prefetch_threads = []

        self.shuffle = shuffle
        self.last_log_time = 0

        assert self.obs_df.soma_joinid.nunique() == self.obs_df.shape[0]
        assert self.obs_df.cell_idx.nunique() == self.obs_df.shape[0]

    def __len__(self):
        return self._len
    
    
    def _calc_start_end(self, worker_id):
        # we have less blocks than workers
        if self.num_blocks < self.num_workers:
            # change num_blocks and block_size
            self.num_blocks = self.num_workers
            self.block_size = math.ceil(self.obs_df.shape[0] / self.num_blocks)

            start_block = worker_id
            end_block = worker_id + 1
        else:
            num_blocks_per_worker = math.floor(self.num_blocks / self.num_workers)
            start_block = worker_id * num_blocks_per_worker
            end_block = start_block + num_blocks_per_worker

            if worker_id + 1 == self.num_workers:
                end_block = self.num_blocks

        return (start_block, end_block)
    
    def _prefetch_worker(self, worker_id, start_block, end_block):
        """Worker thread that prefetches data blocks"""
        with self.worker_lock:
            self.active_workers += 1
            
        try:
            # Add monitoring variables
            cells_processed = 0
            cells_queued = 0
            start_time = time.time()
            last_report_time = start_time
            db_fetch_total_time = 0
            processing_total_time = 0
            queue_total_time = 0
            
            for block_idx in range(start_block, end_block):
                block_start_time = time.time()
                
                # Get block start and end indices
                start_idx = block_idx * self.block_size
                end_idx = min(start_idx + self.block_size, self.obs_df.shape[0])
                
                # print(f"Worker {worker_id} prefetching block {block_idx} ({start_idx}-{end_idx})")
                
                # Get block data
                obs_df_block = self.obs_df.iloc[start_idx:end_idx, :]
                
                cell_idx_block = obs_df_block['cell_idx'].to_numpy(dtype=np.int64)
                soma_joinid_block = obs_df_block["soma_joinid"].to_numpy(dtype=np.int64)
                
                modality_idx_block = obs_df_block["modality_idx"].to_numpy() # TODO: implement for predict mode, always need modality to route to right encoder
                study_idx_block = obs_df_block["study_idx"].to_numpy() if not self.predict_mode else None
                sample_idx_block = obs_df_block["sample_idx"].to_numpy()
                
                # Fetch X data from database with lock to prevent concurrent access
                db_start_time = time.time()
                with self.db_lock:
                    try:
                        with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                            with soma_experiment.axis_query("RNA", obs_query=soma.AxisQuery(coords=(tuple(soma_joinid_block),))) as query:
                                adata = query.to_anndata(X_name='row_raw', column_names={"obs":["soma_joinid"], "var":[]})
                                adata.obs_names = adata.obs["soma_joinid"].astype(str)
                                
                                # Make soma_joinid_block a list of strings
                                soma_joinid_block_str = [str(x) for x in soma_joinid_block]
                                adata = adata[soma_joinid_block_str, :]
                                
                                assert np.all(adata.obs["soma_joinid"] == soma_joinid_block)
                                
                                X_block = adata.X
                        
                        X_block = X_block[:, self.genes_to_use]
                    except Exception as error:
                        print(f"Error reading X array of block {block_idx}: {error}")
                        raise ValueError()
                db_fetch_time = time.time() - db_start_time
                db_fetch_total_time += db_fetch_time
                
                # print(f"Worker {worker_id} - Block {block_idx} DB fetch completed in {db_fetch_time:.2f}s")
                
                # Process each cell in the block and add to queue
                for i in range(len(soma_joinid_block)):
                    # Extract data for this cell
                    processing_start_time = time.time()
                    X_curr = np.squeeze(np.transpose(X_block[i, :].toarray()))
                    
                    if self.pretrained_gene_indices is not None:
                        # Expand X_curr to full size of pretrained model
                        X_curr_full = np.zeros(self.num_input, dtype=np.int32)
                        X_curr_full[self.pretrained_gene_indices] = X_curr
                        X_curr = np.squeeze(np.transpose(X_curr_full))
                    
                    soma_joinid = soma_joinid_block[i]
                    cell_idx = cell_idx_block[i]
                    sample_idx_curr = sample_idx_block[i]
                    feature_presence_mask = self.feature_presence_matrix[sample_idx_curr, :]

                    modality_idx_curr = modality_idx_block[i]
                    
                    # Create datum dictionary based on mode
                    if self.predict_mode:
                        datum = {
                            "x": torch.from_numpy(X_curr.astype("int32")),
                            "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
                            "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
                            "feature_presence_mask": torch.from_numpy(feature_presence_mask),
                            "batch_idx": torch.tensor([modality_idx_curr, -1, -1], dtype=torch.int64),
                
                        }
                    else:
                        study_idx_curr = study_idx_block[i]
                        
                        # Library calculations
                        if sample_idx_curr in self.library_calcs.index:
                            local_l_mean = self.library_calcs.loc[sample_idx_curr, "library_log_means"]
                            local_l_var = self.library_calcs.loc[sample_idx_curr, "library_log_vars"]
                        else:
                            local_l_mean = 0.0
                            local_l_var = 1.0
                        
                        datum = {
                            "x": torch.from_numpy(X_curr.astype("int32")),
                            "soma_joinid": torch.tensor(soma_joinid, dtype=torch.int64),
                            "cell_idx": torch.tensor(cell_idx, dtype=torch.int64),
                            "feature_presence_mask": torch.from_numpy(feature_presence_mask),  
                            "batch_idx": torch.tensor([modality_idx_curr, study_idx_curr, sample_idx_curr], dtype=torch.int64),
                            "local_l_mean": torch.tensor(local_l_mean),
                            "local_l_var": torch.tensor(local_l_var),
                        }
                    processing_time = time.time() - processing_start_time
                    processing_total_time += processing_time
                    
                    # Add to queue (will block if queue is full)
                    queue_start_time = time.time()
                    queue_size_before = self.data_queue.qsize()
                    self.data_queue.put(datum)
                    queue_size_after = self.data_queue.qsize()
                    queue_time = time.time() - queue_start_time
                    queue_total_time += queue_time
                    
                    cells_processed += 1
                    cells_queued += 1
                    
                    # # Check if queue size changed as expected
                    # if queue_size_after != queue_size_before + 1:
                    #     print(f"Worker {worker_id} - Queue anomaly: Size before put: {queue_size_before}, after: {queue_size_after}")
                    
                
                # block_total_time = time.time() - block_start_time
                # print(f"Worker {worker_id} - Block {block_idx} completed: "
                #       f"DB fetch: {db_fetch_time:.2f}s, Total: {block_total_time:.2f}s, "
                #       f"Cells: {len(soma_joinid_block)}, "
                #       f"Current queue size: {self.data_queue.qsize()}")
                
                # # Report overall statistics periodically
                # current_time = time.time()
                # if current_time - last_report_time > 30:  # Report every 30 seconds
                #     elapsed = current_time - start_time
                #     print(f"Worker {worker_id} STATS - "
                #           f"Processed {cells_processed} cells in {elapsed:.2f}s "
                #           f"({cells_processed / elapsed:.2f} cells/sec). "
                #           f"DB time: {db_fetch_total_time:.2f}s ({db_fetch_total_time/elapsed*100:.1f}%), "
                #           f"Processing time: {processing_total_time:.2f}s ({processing_total_time/elapsed*100:.1f}%), "
                #           f"Queue time: {queue_total_time:.2f}s ({queue_total_time/elapsed*100:.1f}%)")
                #     last_report_time = current_time
            
        finally:
            with self.worker_lock:
                self.active_workers -= 1
                if self.active_workers == 0 and not self.data_queue.full():
                    self.data_queue.put(None)
                    
            # total_time = time.time() - start_time
            # print(f"Worker {worker_id} FINISHED - Processed {cells_processed} cells in {total_time:.2f}s "
            #       f"({cells_processed / total_time:.2f} cells/sec). "
            #       f"DB time: {db_fetch_total_time:.2f}s ({db_fetch_total_time/total_time*100:.1f}%), "
            #       f"Processing time: {processing_total_time:.2f}s ({processing_total_time/total_time*100:.1f}%), "
            #       f"Queue time: {queue_total_time:.2f}s ({queue_total_time/total_time*100:.1f}%)")
    
    def __iter__(self):
        # Initialize thread-related objects here to avoid pickling issues
        self.data_queue = queue.Queue(maxsize=self.max_queue_size)
        self.db_lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.active_workers = 0
        self.prefetch_threads = []
        
        if torch.utils.data.get_worker_info():
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id
            
            # Calculate block range for this worker
            self.start_block, self.end_block = self._calc_start_end(self.worker_id)
        else:
            self.worker_id = 0
            self.start_block = 0
            self.end_block = self.num_blocks
        
        # Start prefetch thread
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(self.worker_id, self.start_block, self.end_block),
            daemon=True
        )
        thread.start()
        self.prefetch_threads.append(thread)
        
        if self.verbose:
            print(f"Worker {self.worker_id} - start block: {self.start_block}, end block: {self.end_block}, block_size: {self.block_size}")

        if self.shuffle:
            self.obs_df = self.obs_df.sample(frac=1).reset_index(drop=True)
        
        
        return self
    
    def __next__(self):
        # Add monitoring for consumption rate
        start_time = time.time()
        
        # Get next item from queue
        datum = self.data_queue.get()
        
        # Check if we've reached the end
        if datum is None:
            raise StopIteration
        
        get_time = time.time() - start_time
        
        # if queue size is less than 5% of max size, log
        # and been 15 seconds since last log
        if self.data_queue.qsize() < self.max_queue_size * 0.05 and time.time() - self.last_log_time > 15:
            print(f"Worker {self.worker_id} - Queue size: {self.data_queue.qsize()}, max size: {self.max_queue_size}")
            self.last_log_time = time.time()
        
        
        return datum
            



