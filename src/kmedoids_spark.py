import sys
import os
import time
from statistics import median
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from common import TABLES, load_df, upsert
from kmedoids import SEED, encode, hamming_matrix, kmedoids

K = 3
SIZES = [2000, 5000, 10000, 20000]
R = int(os.environ.get("REPEATS", 3))


def dist_to(x, rows):
    d = np.zeros((len(x), len(rows)), dtype=np.float32)
    for j in range(x.shape[1]):
        d += x[:, j][:, None] != rows[:, j][None, :]
    return d / x.shape[1]


def init_pp(x, k, rng):
    med = [rng.integers(len(x))]
    for _ in range(1, k):
        dist = dist_to(x, x[med]).min(axis=1)
        p = dist / dist.sum() if dist.sum() > 0 else None
        med.append(rng.choice(len(x), p=p))
    return np.array(med)


def assign(med_b):
    def f(it):
        for off, xs in it:
            d = dist_to(xs, med_b.value)
            yield off, d.argmin(axis=1), d.min(axis=1)
    return f


def update(x_b, lab_b, k):
    def f(it):
        x, lab = x_b.value, lab_b.value
        best = {}
        for off, xs in it:
            for i in range(len(xs)):
                g = off + i
                c = lab[g]
                idx = np.where(lab == c)[0]
                s = dist_to(xs[i:i + 1], x[idx])[0].sum()
                if c not in best or s < best[c][0] or (s == best[c][0] and g < best[c][1]):
                    best[c] = (s, g)
        return list(best.items())
    return f


def merge(a, b):
    return a if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]) else b


def run_assign(sc, chunks, med):
    mb = sc.broadcast(med)
    res = sorted(chunks.mapPartitions(assign(mb)).collect(), key=lambda r: r[0])
    mb.unpersist()
    return np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res])


def kmedoids_spark(sc, x, chunks, k, rng, n_init=5, max_iter=100):
    xb = sc.broadcast(x)
    best = None
    for _ in range(n_init):
        med = init_pp(x, k, rng)
        for _ in range(max_iter):
            lab, _ = run_assign(sc, chunks, x[med])
            lb = sc.broadcast(lab)
            res = chunks.mapPartitions(update(xb, lb, k)).reduceByKey(merge).collectAsMap()
            lb.unpersist()
            new = med.copy()
            for c, (_, g) in res.items():
                new[c] = g
            if set(new) == set(med):
                break
            med = new
        lab, mind = run_assign(sc, chunks, x[med])
        cost = mind.sum()
        if best is None or cost < best[0]:
            best = (cost, med, lab)
    xb.unpersist()
    return best


if __name__ == "__main__":
    t0 = time.perf_counter()
    spark = SparkSession.builder.master("local[*]").appName("kmedoids").config("spark.driver.memory", "6g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    session_s = time.perf_counter() - t0
    sc = spark.sparkContext
    par = sc.defaultParallelism
    full = load_df()
    rows, ver = [], []
    for n in [int(a) for a in sys.argv[1:]] or SIZES:
        df = full.sample(n, random_state=SEED, replace=n > len(full)).reset_index(drop=True)
        pt = []
        for _ in range(R):
            t0 = time.perf_counter()
            x = encode(df)
            d = hamming_matrix(x)
            pc, pm, _ = kmedoids(d, K, np.random.default_rng(SEED))
            pt.append(time.perf_counter() - t0)
            del d
        py_s = median(pt)
        st = []
        for _ in range(R):
            t0 = time.perf_counter()
            x = encode(df)
            step = -(-n // par)
            chunks = sc.parallelize([(o, x[o:o + step]) for o in range(0, n, step)], par).cache()
            sc_, sm, _ = kmedoids_spark(sc, x, chunks, K, np.random.default_rng(SEED))
            st.append(time.perf_counter() - t0)
            chunks.unpersist()
        sp_s = median(st)
        rows.append({"algorithm": "K-Medoids", "size": n, "python_s": round(py_s, 4), "spark_s": round(sp_s, 4), "repeats": R})
        ver.append({"algorithm": "K-Medoids", "size": n, "python_result": " ".join(map(str, sorted(pm))), "spark_result": " ".join(map(str, sorted(sm))), "match": sorted(pm) == sorted(sm)})
        print(rows[-1], ver[-1], round(float(pc), 3), round(float(sc_), 3), flush=True)
    upsert(TABLES / "spark_kmedoids_runs.csv", pd.DataFrame(rows), ["size"])
    upsert(TABLES / "spark_kmedoids_verify.csv", pd.DataFrame(ver), ["size"])
    upsert(TABLES / "spark_session_kmedoids.csv", pd.DataFrame([{"script": "kmedoids_spark", "sizes": " ".join(str(r["size"]) for r in rows), "session_start_s": round(session_s, 4)}]), ["script"])
    spark.stop()
