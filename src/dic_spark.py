import sys
import os
import time
from statistics import median
from collections import Counter
from itertools import combinations
import pandas as pd
from pyspark.sql import SparkSession
from common import TABLES, MIN_SUPPORT, ATTRS, load_transactions, upsert
from dic import dic, DC, DB, SC, SB
from scale import SCALES, scaled_path

BASE_BLOCK = 2000
R = int(os.environ.get("REPEATS", 3))


def count_block(active_b, maxlen):
    def f(it):
        active = active_b.value
        c = Counter()
        for t in it:
            st = sorted(t)
            for r in range(1, min(maxlen, len(st)) + 1):
                for sub in combinations(st, r):
                    fs = frozenset(sub)
                    if fs in active:
                        c[fs] += 1
        return list(c.items())
    return f


def dic_spark(sc, rdd, n, nblocks, par, min_support=MIN_SUPPORT):
    minc = min_support * n
    sizes = rdd.glom().map(len).collect()
    bsizes = [sum(sizes[i * par:(i + 1) * par]) for i in range(nblocks)]
    items = sorted(rdd.flatMap(lambda t: t).distinct().collect())
    state = {frozenset([i]): DC for i in items}
    count = {k: 0 for k in state}
    seen = {k: 0 for k in state}
    boxes = set()
    b = 0
    while any(s in (DC, DB) for s in state.values()):
        active = {k for k, s in state.items() if s in (DC, DB)}
        maxlen = max(len(k) for k in active)
        ab = sc.broadcast(active)
        for k, v in sc.runJob(rdd, count_block(ab, maxlen), partitions=list(range(b * par, (b + 1) * par))):
            count[k] += v
        ab.unpersist()
        for k in active:
            seen[k] += bsizes[b]
        new_boxes = []
        for k in active:
            if state[k] == DC and count[k] >= minc:
                state[k] = DB
                boxes.add(k)
                new_boxes.append(k)
        singles = [k for k in boxes if len(k) == 1]
        for x in new_boxes:
            for y in singles:
                c = x | y
                if len(c) == len(x) + 1 and c not in state and all(c - {z} in boxes for z in c):
                    state[c] = DC
                    count[c] = 0
                    seen[c] = 0
        for k in active:
            if seen[k] >= n:
                state[k] = SB if state[k] == DB else SC
        b = (b + 1) % nblocks
    return {k: count[k] for k, s in state.items() if s == SB}, n


if __name__ == "__main__":
    t0 = time.perf_counter()
    spark = SparkSession.builder.master("local[*]").appName("dic").config("spark.driver.memory", "6g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    session_s = time.perf_counter() - t0
    sc = spark.sparkContext
    par = sc.defaultParallelism
    rows, ver = [], []
    for s in [int(a) for a in sys.argv[1:]] or SCALES:
        pdf = pd.read_csv(scaled_path(s))
        tx = load_transactions(pdf)
        n = len(tx)
        block = BASE_BLOCK * s
        nblocks = -(-n // block)
        pt = []
        for _ in range(R):
            t0 = time.perf_counter()
            pf, _ = dic(tx, block=block)
            pt.append(time.perf_counter() - t0)
        py_s = median(pt)
        sdf = spark.read.csv(str(scaled_path(s)), header=True).select(*ATTRS)
        rdd = sdf.rdd.zipWithIndex().map(lambda r: (r[1] // block * par + r[1] % par, frozenset(f"{a}={v}" for a, v in zip(ATTRS, r[0])))).partitionBy(nblocks * par, lambda k: k).values().cache()
        rdd.count()
        st = []
        for _ in range(R):
            t0 = time.perf_counter()
            sf, _ = dic_spark(sc, rdd, n, nblocks, par)
            st.append(time.perf_counter() - t0)
        sp_s = median(st)
        rdd.unpersist()
        rows.append({"algorithm": "DIC", "size": n, "python_s": round(py_s, 4), "spark_s": round(sp_s, 4), "repeats": R})
        ver.append({"algorithm": "DIC", "size": n, "python_result": len(pf), "spark_result": len(sf), "match": pf == sf})
        print(rows[-1], ver[-1], flush=True)
    upsert(TABLES / "spark_dic_runs.csv", pd.DataFrame(rows), ["size"])
    upsert(TABLES / "spark_dic_verify.csv", pd.DataFrame(ver), ["size"])
    upsert(TABLES / "spark_session_dic.csv", pd.DataFrame([{"script": "dic_spark", "sizes": " ".join(str(r["size"]) for r in rows), "session_start_s": round(session_s, 4)}]), ["script"])
    spark.stop()
