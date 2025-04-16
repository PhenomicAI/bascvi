import threading
from concurrent.futures import ThreadPoolExecutor
import os
import wandb

from ml_benchmarking.bascvi.utils.utils import umap_calc_and_plot

class AsyncUMAPGenerator:
    def __init__(self, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks = {}
        self.lock = threading.Lock()
        
    def submit_umap_task(self, key_prefix, embeddings_df, emb_columns, save_dir, obs_df, obs_columns, 
                          epoch=None, step=None, max_cells=100000, opacity=0.3, **kwargs):
        """Submit a UMAP generation task asynchronously"""
        with self.lock:
            # Create a unique task key that includes the epoch
            task_key = f"{key_prefix}_epoch{epoch}" if epoch is not None else key_prefix
            
            # Don't submit if we already have a pending task with the same key
            if task_key in self.pending_tasks:
                return
                
            future = self.executor.submit(
                self._generate_and_log_umap, 
                key_prefix, embeddings_df, emb_columns, save_dir, obs_df, obs_columns, 
                epoch, step, max_cells, opacity, **kwargs
            )
            self.pending_tasks[task_key] = future
        
    def _generate_and_log_umap(self, key_prefix, embeddings_df, emb_columns, save_dir, obs_df, obs_columns, 
                               epoch, step, max_cells=100000, opacity=0.3, **kwargs):
        """Generate UMAP in background thread and log directly to wandb"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"Generating UMAP for {key_prefix} (epoch {epoch})...")
            _, _, fig_path_dict = umap_calc_and_plot(
                embeddings_df, emb_columns, save_dir, obs_columns, 
                max_cells=max_cells, opacity=opacity, **kwargs
            )
            
            # Log directly to wandb
            if fig_path_dict:
                wandb_images = {}
                for key, fig_path in fig_path_dict.items():
                    # Create proper wandb key
                    full_key = f"{key_prefix}/{key}" if "/" not in key else key
                    wandb_images[full_key] = wandb.Image(fig_path)
                
                # Log with step if provided
                if wandb.run is not None:
                    log_dict = {
                        **wandb_images,
                        "epoch": epoch if epoch is not None else 0
                    }
                    if step is not None:
                        wandb.log(log_dict, step=step)
                    else:
                        wandb.log(log_dict)
                    
                    print(f"Logged {len(wandb_images)} UMAP images to wandb for {key_prefix} (epoch {epoch})")
                
            return True
        except Exception as e:
            import traceback
            print(f"UMAP generation failed for {key_prefix}: {e}")
            print(traceback.format_exc())
            return False
            
    def check_pending(self):
        """Report status of pending tasks and clean up completed ones"""
        with self.lock:
            # Check all pending tasks and remove completed ones
            keys_to_remove = []
            for task_key, future in self.pending_tasks.items():
                if future.done():
                    try:
                        # Just check if there was an exception
                        future.result()
                    except Exception as e:
                        print(f"Error in UMAP task {task_key}: {e}")
                    finally:
                        keys_to_remove.append(task_key)
            
            # Remove completed/failed tasks
            for key in keys_to_remove:
                self.pending_tasks.pop(key, None)
            
            return len(self.pending_tasks)