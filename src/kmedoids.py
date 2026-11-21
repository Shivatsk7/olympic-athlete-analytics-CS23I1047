import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from common import TABLES, ATTRS, load_df, record_time

SEED = 42
SAMPLE = 5000
KS = range(2, 9)


def encode(df):
    return np.stack([pd.factorize(df[a])[0] for a in ATTRS], axis=1)


def hamming_matrix(x):
    d = np.zeros((len(x), len(x)), dtype=np.float32)
    for j in range(x.shape[1]):
        d += x[:, j][:, None] != x[:, j][None, :]
    d /= x.shape[1]
    return d


def init_pp(d, k, rng):
    med = [rng.integers(len(d))]
    for _ in range(1, k):
        dist = d[:, med].min(axis=1)
        p = dist / dist.sum() if dist.sum() > 0 else None
        med.append(rng.choice(len(d), p=p))
    return np.array(med)


def kmedoids(d, k, rng, n_init=5, max_iter=100):
    best = None
    for _ in range(n_init):
        med = init_pp(d, k, rng)
        for _ in range(max_iter):
            lab = d[:, med].argmin(axis=1)
            new = med.copy()
            for c in range(k):
                idx = np.where(lab == c)[0]
                if len(idx):
                    new[c] = idx[d[np.ix_(idx, idx)].sum(axis=1).argmin()]
            if set(new) == set(med):
                break
            med = new
        lab = d[:, med].argmin(axis=1)
        cost = d[np.arange(len(d)), med[lab]].sum()
        if best is None or cost < best[0]:
            best = (cost, med, lab)
    return best


if __name__ == "__main__":
    df = load_df().sample(SAMPLE, random_state=SEED).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    x = encode(df)
    d = hamming_matrix(x)
    scores, fits = [], {}
    for k in KS:
        cost, med, lab = kmedoids(d, k, rng)
        sil = silhouette_score(d, lab, metric="precomputed")
        scores.append({"k": k, "silhouette": round(float(sil), 5), "cost": round(float(cost), 3)})
        fits[k] = (med, lab)
    el = time.perf_counter() - t0
    sc = pd.DataFrame(scores)
    sc.to_csv(TABLES / "kmedoids_k_scores.csv", index=False)
    bk = int(sc.loc[sc["silhouette"].idxmax(), "k"])
    med, lab = fits[bk]
    cl = df.loc[med, ATTRS].reset_index(drop=True)
    cl.insert(0, "cluster", range(bk))
    cl["size"] = np.bincount(lab, minlength=bk)
    cl.to_csv(TABLES / "kmedoids_clusters.csv", index=False)
    prof = []
    for c in range(bk):
        sub = df.loc[lab == c, ATTRS]
        for at in ATTRS:
            vc = sub[at].value_counts(normalize=True)
            prof.append({"cluster": c, "attribute": at, "top_value": vc.index[0], "share": round(float(vc.iloc[0]), 4)})
    pd.DataFrame(prof).to_csv(TABLES / "kmedoids_cluster_profiles.csv", index=False)
    record_time("K-Medoids", "python", el, SAMPLE)
    print(sc.to_string(index=False))
    print(f"best k={bk}, {el:.3f}s")
    print(cl.to_string(index=False))
