# BA-scVI on scMARK — reproducible benchmark

## Released weights

| benchmark | weights | KNI |
|---|---|---|
| **scMARK** | [socooper/bascvi-scmark](https://huggingface.co/socooper/bascvi-scmark) — `bascvi_scmark_epoch63.ckpt` | **0.7155** |
| **scREF / organism-wide** | [phenomicai/bascvi-human](https://huggingface.co/phenomicai/bascvi-human) — `human_bascvi_epoch_123.ckpt` | paper reports 0.632 on scREF |

The scMARK checkpoint embeds its own `gene_list` (35,804 genes) in
`hyper_parameters`, so it is self-contained. The scREF/human checkpoint uses a
different 29,494-gene space and 3,019 batch categories, and scores 0.582 on
scMARK — it is a different model, not the scMARK one, so do not use it to
reproduce scMARK numbers.

**Note on loading `bascvi-human`:** it predates the per-batch-level
`z_predictors`/`x_predictors` ModuleList and has a single `z_predictor`, so it
does **not** load with current `BAScVI`. Commit `b411756` is the last one whose
model matches it.

## Quick start

```bash
pip install -e .
python src/ml_benchmarking/scripts/scmark_benchmark.py --work-dir ./scmark_run
```

Downloads scMARK v2 (Zenodo, 400 MB) and the checkpoint (HuggingFace), embeds all
109,543 cells and prints the KNI score. Nothing else is required.

Expected output:

```
scMARK KNI = 0.7155   (cross-study acc 0.7184, batch diversity 0.995)
published BA-scVI reference: 0.7110
```

## Changes in this branch

**`adv_loss_mode="confusion"` (new, used by the released weights).** The default
`neg_ce` objective has the generator *maximise* the discriminator's cross-entropy.
That is unbounded above and its optimum is not a batch-free representation: pushing
CE past log(K) makes the discriminator confidently *wrong*, i.e. batch identity is
still linearly present, merely sign-flipped. With a large weight the discriminator's
only stable response is to stop depending on its input at all. The confusion
objective instead targets the uniform posterior, which is bounded below by log(K)
and attained exactly when the representation is uninformative about batch.

**Per-level normalisation in `get_disc_loss`.** Batch levels differ enormously in
class count (modality 2, study 11, sample 354 on scMARK; up to ~8,000 samples on
larger corpora), so raw cross-entropies span log(2)=0.69 to log(8308)=9.03 and
averaging them lets the largest level dominate the gradient. Each level is now
divided by log(K), putting all of them on a common "1.0 == chance" scale.

**Single-class levels are skipped.** A level with one class has a constant softmax,
contributes no gradient, and only dilutes the mean.

**`bascvi/utils/kni_fast.py` (new).** Vectorised KNI, equivalent to
`utils.calc_kni_score` but fast enough to run as a per-epoch validation metric
(the reference implementation loops over every cell in Python). Verified to agree
with the original on the authors' released embedding: 0.7110 vs 0.7114.

## Notes for reproduction

- KNI is computed over all cells, including those the model trained on. This is the
  published benchmark's own methodology and applies equally to the 0.7110 reference.
- KNI rewards low-dimensional embeddings via its batch-diversity gate; all numbers
  here are at 10-d.
- Training used batch injection into the **decoder only**; the encoder never sees
  batch, and `predict_mode=True` zeroes the batch vector at inference.


## Known issues encountered while reproducing

These were hit while training BA-scVI on a corpus built from scratch (the
released S3 corpus has the required arrays baked in, so these paths are rarely
exercised). Not all are fixed in this branch — listing them so others do not
lose time:

1. **`filter_and_generate_library_calcs()` uses `filter_pass_ids` before it is
   assigned**, so it raises `AttributeError` on any corpus lacking a
   `sample_library_calcs` array. Workaround: precompute
   `ms["RNA"]["sample_library_calcs"]` (columns `sample_idx`,
   `library_log_means`, `library_log_vars`) before training.
2. **`ms["RNA"]["feature_presence_matrix"]` is required** but not created by any
   script in the repo. It is `[n_samples, n_genes]`, 1 where that sample's study
   measured the gene. Presence should come from the study's gene panel, not from
   expression non-zeros — genes a study never assayed are structural zeros and
   must be masked out of the reconstruction loss.
3. **`obs` must contain `sample_name`** or the post-training embedding export
   crashes *after* training completes, losing the run's output.
4. **`AnnDataDataset` metadata is wrong with `num_workers > 1`.** `file_path`,
   `file_counter` and `cell_counter` are instance attributes mutated during
   iteration, so each worker's copy diverges and the emitted values do not
   correspond to the row's actual source. In one 11-file run, one study was
   credited with 79,540 rows against its true 9,540 and only 4 of 11 files
   appeared. This silently corrupts any join of embeddings back to cell
   metadata — including the one in the tutorial notebook. Use `num_workers=1`
   until fixed.
5. **Python 3.12+**: `import imp` (removed) in `bascvi/model/__init__.py`, and
   `AnnData(dtype=...)` (removed from anndata) in
   `datamodule/anndata/dataset.py`. The first is fixed here.
