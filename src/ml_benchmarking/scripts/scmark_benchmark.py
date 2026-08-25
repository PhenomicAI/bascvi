"""Reproduce the scMARK benchmark result for the released BA-scVI weights.

Downloads scMARK v2 from Zenodo, maps it into the model's gene space, embeds
every cell, and scores the K-Neighbors Intersection (KNI) metric.

    python scmark_benchmark.py --work-dir ./scmark_run

Everything is fetched automatically; no private data or credentials required.
"""
import argparse, glob, os, subprocess, sys, tarfile
import numpy as np

ZENODO = "https://zenodo.org/records/7795653/files/scmark_v2.tar.bz2?download=1"
HF_REPO = "socooper/bascvi-scmark"
CKPT = "bascvi_scmark_epoch63.ckpt"


def fetch(work):
    os.makedirs(work, exist_ok=True)
    tgz = os.path.join(work, "scmark_v2.tar.bz2")
    if not os.path.isdir(os.path.join(work, "scmark_v2")):
        if not os.path.exists(tgz):
            print("downloading scMARK v2 (400 MB) ...", flush=True)
            subprocess.check_call(["curl", "-sL", "-o", tgz, ZENODO])
        print("extracting ...", flush=True)
        with tarfile.open(tgz, "r:bz2") as t:
            t.extractall(work)
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download(repo_id=HF_REPO, filename=CKPT, local_dir=work)
    return os.path.join(work, "scmark_v2"), ck


def build(h5ad_dir, gene_list):
    """scMARK -> a matrix in the model's gene order, zero-filling absent genes."""
    import h5py, scipy.sparse as sp
    gidx = {g: i for i, g in enumerate(gene_list)}
    blocks, ct, st = [], [], []
    for p in sorted(glob.glob(os.path.join(h5ad_dir, "*.h5ad"))):
        with h5py.File(p, "r") as f:
            genes = [x.decode() for x in f["var/gene"][:]]
            n = f["obs/index"].shape[0]
            X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                              shape=(n, len(genes)))
            cats = [x.decode() for x in f["obs/__categories/standard_true_celltype"][:]]
            ct.append(np.array(cats, dtype=object)[f["obs/standard_true_celltype"][:]])
            scat = [x.decode() for x in f["obs/__categories/study_name"][:]]
            st.append(np.array(scat, dtype=object)[f["obs/study_name"][:]])
        col = np.array([gidx.get(g, -1) for g in genes])
        keep = col >= 0
        X = X[:, keep]
        blocks.append(sp.csr_matrix((X.data, col[keep][X.indices], X.indptr),
                                    shape=(X.shape[0], len(gene_list))))
        print(f"  {os.path.basename(p)[:44]:46s} {X.shape[0]:6d} cells, "
              f"{keep.sum()}/{len(genes)} genes mapped", flush=True)
    X = sp.vstack(blocks).tocsr()
    X.sum_duplicates()
    return X, np.concatenate(ct), np.concatenate(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="./scmark_run")
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import torch
    from ml_benchmarking.bascvi.model.bascvi import BAScVI
    from ml_benchmarking.bascvi.utils.kni_fast import kni_score, filter_unknown

    h5ad_dir, ck_path = fetch(a.work_dir)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    gene_list = ck["hyper_parameters"]["gene_list"]
    model = BAScVI(**ck["hyper_parameters"]["model_args"])
    model.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items()
                           if k.startswith("vae.")}, strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)

    X, ct, st = build(h5ad_dir, gene_list)
    print(f"\n{X.shape[0]} cells x {X.shape[1]} genes", flush=True)
    d = ck["hyper_parameters"]["model_args"]["n_latent"]
    Z = np.zeros((X.shape[0], d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, X.shape[0], a.batch_size):
            e = min(s + a.batch_size, X.shape[0])
            xb = torch.from_numpy(np.asarray(X[s:e].todense(), dtype=np.float32)).to(dev)
            out = model({"x": xb}, encode=True, predict_mode=True)
            qz = out[0]["qz_m"] if isinstance(out, tuple) else out["qz_m"]
            Z[s:e] = qz.float().cpu().numpy()

    import pandas as pd
    mask, ctf, stf, Zf = filter_unknown(ct, st, Z)
    code = lambda v: pd.Categorical(v).codes
    r = kni_score(np.ascontiguousarray(Zf, dtype=np.float32), code(ctf), code(stf),
                  n_neighbours=50, max_prop_same_batch=0.8, exact_max_cells=200_000)
    print(f"\nscMARK KNI = {r['kni']:.4f}   (cross-study acc {r['acc_cross']:.4f}, "
          f"batch diversity {r['diverse_frac']:.3f})")
    print("published BA-scVI reference: 0.7110")


if __name__ == "__main__":
    main()
