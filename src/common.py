import pandas as pd
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
ATTRS = ["Sport", "Gender", "NOC", "Medal", "Era"]
MIN_SUPPORT = 0.01
MIN_CONF = 0.5


def load_df():
    return pd.read_csv(ROOT / "data" / "clean.csv")


def load_transactions(df=None):
    df = load_df() if df is None else df
    return [frozenset(f"{a}={v}" for a, v in zip(ATTRS, row)) for row in df[ATTRS].itertuples(index=False)]


def gen_rules(supports, n):
    rows = []
    for items, cnt in supports.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in combinations(sorted(items), r):
                a = frozenset(ante)
                c = items - a
                conf = cnt / supports[a]
                if conf >= MIN_CONF:
                    rows.append(rule_row(a, c, cnt / n, conf, supports[c] / n))
    return rules_df(rows)


def rule_row(a, c, sup, conf, sup_c):
    return {"antecedent": ", ".join(sorted(a)), "consequent": ", ".join(sorted(c)), "support": round(sup, 5), "confidence": round(conf, 5), "lift": round(conf / sup_c, 5)}


def rules_df(rows):
    return pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift"]).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def itemsets_df(supports, n):
    rows = [{"itemset": ", ".join(sorted(k)), "size": len(k), "count": v, "support": round(v / n, 5)} for k, v in supports.items()]
    return pd.DataFrame(rows).sort_values(["size", "support"], ascending=[True, False]).reset_index(drop=True)


def record_time(algo, impl, seconds, rows, path=TABLES / "python_timings.csv"):
    new = pd.DataFrame([{"algorithm": algo, "impl": impl, "rows": rows, "seconds": round(seconds, 4)}])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~((old["algorithm"] == algo) & (old["impl"] == impl) & (old["rows"] == rows))], new])
    new.to_csv(path, index=False)


def upsert(path, df, keys):
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old.merge(df[keys], on=keys, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge"), df], ignore_index=True)
    df.to_csv(path, index=False)
