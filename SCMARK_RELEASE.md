# BA-scVI on scMARK — reproducible benchmark

Released weights: **[socooper/bascvi-scmark](https://huggingface.co/socooper/bascvi-scmark)**

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
