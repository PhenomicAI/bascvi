"""Vectorised KNI score, fast enough to run as a per-epoch validation metric.

The reference implementation in ``utils.calc_kni_score`` loops over every cell
in Python, which costs minutes at the scale we want to validate on. This
computes the same quantities with array ops so it can run every epoch.

KNI (K-Neighbors Intersection, arXiv:2503.20730): for each cell, look at its k
nearest neighbours in the embedding. A cell *passes* if its neighbourhood is
batch-diverse (fewer than ``max_prop_same_batch`` of neighbours share its batch)
AND the majority cell-type vote among its **out-of-batch** neighbours matches
its own label. One score that penalises both batch effects (via the diversity
requirement) and loss of biological signal (via label prediction).

Verified against calc_kni_score in tests/test_core/test_kni_fast.py.
"""
from __future__ import annotations

import numpy as np

#: Labels that carry no cell-type information. Cells with these are excluded --
#: they would otherwise be scored as a real class and inflate (or deflate) the
#: result depending on how many there are. Matches calc_kni_score's list.
UNKNOWN_LABELS = frozenset({
    "Unknown", "unknown", "", "nan", "NaN", "NA", "na", "N/A", "n/a", "None", "null",
})


def scale_embeddings(X: np.ndarray) -> np.ndarray:
    """Centre each dimension and scale by its IQR.

    Matches ``utils.scale_embeddings``: without it, dimensions with larger
    spread dominate the Euclidean neighbourhood, so the score would depend on
    the arbitrary scale of individual latent dimensions.
    """
    X = np.asarray(X, dtype=np.float32).copy()
    X -= X.mean(axis=0, keepdims=True)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    flat = iqr == 0
    X[:, ~flat] /= iqr[~flat]
    X[:, flat] = 0.0
    return X


def filter_unknown(cell_type_labels, *arrays, unknown=UNKNOWN_LABELS):
    """Drop cells whose cell-type label is uninformative.

    Returns (mask, filtered_labels, *filtered_arrays). Apply before coding
    labels to integers so the code space contains only real classes.
    """
    lab = np.asarray(cell_type_labels, dtype=object)
    mask = np.array([(l is not None) and (str(l) not in unknown) for l in lab], dtype=bool)
    return (mask, lab[mask]) + tuple(np.asarray(a)[mask] for a in arrays)


def _majority_vote(rows: np.ndarray, labels: np.ndarray, n_classes: int,
                   n_cells: int) -> np.ndarray:
    """Per-row majority label via a single flat bincount.

    ``rows``/``labels`` are parallel flat arrays of (cell, neighbour-label)
    pairs. Encoding them as ``row * n_classes + label`` turns the per-row
    tally into one bincount, which is what makes this fast.
    """
    flat = rows.astype(np.int64) * n_classes + labels.astype(np.int64)
    votes = np.bincount(flat, minlength=n_cells * n_classes)
    return votes.reshape(n_cells, n_classes).argmax(axis=1)


def kni_score(
    embeddings: np.ndarray,
    cell_type: np.ndarray,
    batch: np.ndarray,
    n_neighbours: int = 50,
    max_prop_same_batch: float = 0.8,
    use_faiss: bool | None = None,
    scale: bool = True,
    exact_max_cells: int = 100_000,
    nprobe: int = 32,
) -> dict:
    """Compute the KNI score.

    Parameters
    ----------
    embeddings
        ``(n_cells, n_dims)`` latent coordinates.
    cell_type, batch
        Integer-coded labels, length ``n_cells``. Code them once outside
        (e.g. ``pd.factorize``) so repeated calls stay cheap.
    n_neighbours
        k for the neighbourhood (excludes the cell itself).
    max_prop_same_batch
        A neighbourhood is "diverse" if fewer than this proportion of its
        neighbours share the cell's batch.
    use_faiss
        Force faiss on/off. Default: use it when importable, since exact
        sklearn kNN dominates runtime at >100k cells.
    scale
        Quantile-scale each latent dimension first (matches the reference).
        Without it the score depends on the arbitrary scale of each dimension.
    exact_max_cells
        Use exact search up to this many cells, an approximate IVF index above
        it. Exact search is quadratic and becomes the dominant cost of a
        validation epoch well before the training step does.
    nprobe
        IVF cells probed per query. Higher = more accurate, slower.

    Cells with uninformative cell-type labels must be removed *before* calling
    this -- see :func:`filter_unknown`.

    Returns
    -------
    dict with ``kni`` (the headline score), ``acc`` (plain kNN label accuracy,
    ignoring batch diversity) and ``diverse_frac`` (fraction of cells whose
    neighbourhood was batch-diverse). Comparing ``kni`` against ``acc`` shows
    how much of the label signal survives the batch-diversity requirement.
    """
    X = scale_embeddings(embeddings) if scale else np.asarray(embeddings, dtype=np.float32)
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    ct = np.asarray(cell_type)
    bt = np.asarray(batch)
    if not (len(ct) == len(bt) == n):
        raise ValueError("embeddings, cell_type and batch must be the same length")

    k = int(min(n_neighbours, max(1, n - 1)))

    if use_faiss is None:
        try:
            import faiss  # noqa: F401
            use_faiss = True
        except ImportError:
            use_faiss = False

    if use_faiss:
        import faiss

        # Exact (IndexFlatL2) search is O(n^2): measured 603s at n=667k versus
        # 17.6s for IVF, at 98.8% neighbour agreement. Above `exact_max_cells`
        # the approximation is the only way this is affordable per epoch.
        if n > exact_max_cells:
            nlist = max(1, int(np.sqrt(n)))
            quantizer = faiss.IndexFlatL2(X.shape[1])
            index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist)
            index.train(X)
            index.add(X)
            index.nprobe = min(nprobe, nlist)
        else:
            index = faiss.IndexFlatL2(X.shape[1])
            index.add(X)
        _, idx = index.search(X, k + 1)
    else:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self-match

    knn_ct = ct[idx]
    knn_bt = bt[idx]

    # Batch diversity: how many neighbours share this cell's batch.
    same_batch = knn_bt == bt[:, None]
    diverse = same_batch.sum(axis=1) < max_prop_same_batch * k

    n_classes = int(ct.max()) + 1
    rows = np.repeat(np.arange(n, dtype=np.int64), k)

    # Plain kNN accuracy over all neighbours.
    pred_all = _majority_vote(rows, knn_ct.ravel(), n_classes, n)
    acc = float(np.mean(pred_all == ct))

    # KNI: majority vote over out-of-batch neighbours only, and only for cells
    # whose neighbourhood was diverse. A cell with a diverse neighbourhood but
    # no out-of-batch neighbours cannot be scored, so it counts as a failure --
    # matching the reference, where bincount on an empty selection would error.
    out = ~same_batch
    keep = diverse[:, None] & out
    sel_rows = rows[keep.ravel()]
    sel_lab = knn_ct.ravel()[keep.ravel()]
    pred_nb = _majority_vote(sel_rows, sel_lab, n_classes, n)
    has_votes = np.zeros(n, dtype=bool)
    has_votes[np.unique(sel_rows)] = True

    # Accuracy split by where the votes came from. `acc` above pools all
    # neighbours, so a model that merely packs each study tightly scores well on
    # it without integrating anything -- it is not the paper's metric and must
    # not be read as one. These two separate the components: `acc_within` is
    # how well a cell is called by its own study, `acc_cross` by every other
    # study, over all cells and with no diversity gate.
    in_rows = rows[same_batch.ravel()]
    in_lab = knn_ct.ravel()[same_batch.ravel()]
    pred_in = _majority_vote(in_rows, in_lab, n_classes, n)
    has_in = np.zeros(n, dtype=bool)
    has_in[np.unique(in_rows)] = True
    acc_within = float(np.mean((pred_in == ct)[has_in])) if has_in.any() else float("nan")

    out_rows = rows[out.ravel()]
    out_lab = knn_ct.ravel()[out.ravel()]
    pred_out = _majority_vote(out_rows, out_lab, n_classes, n)
    has_out = np.zeros(n, dtype=bool)
    has_out[np.unique(out_rows)] = True
    acc_cross = float(np.mean((pred_out == ct)[has_out])) if has_out.any() else float("nan")

    kni_hit = diverse & has_votes & (pred_nb == ct)
    return {
        "acc_within": acc_within,
        "acc_cross": acc_cross,
        "kni": float(kni_hit.sum() / n),
        "acc": acc,
        "diverse_frac": float(diverse.mean()),
        "n_cells": int(n),
        "n_cell_types": int(n_classes),
        "n_batches": int(bt.max()) + 1,
        # Per-cell hit vector, so callers can break the score down by study or
        # cell type without rebuilding the neighbour graph (which would change
        # it -- neighbours must be found in the full embedding, not per group).
        "per_cell_correct": kni_hit,
    }
