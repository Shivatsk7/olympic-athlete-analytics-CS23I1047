import time
from itertools import combinations
from common import TABLES, MIN_SUPPORT, load_transactions, gen_rules, itemsets_df, record_time

DC, DB, SC, SB = "DC", "DB", "SC", "SB"


def dic(transactions, min_support=MIN_SUPPORT, block=2000):
    n = len(transactions)
    minc = min_support * n
    items = sorted({i for t in transactions for i in t})
    state = {frozenset([i]): DC for i in items}
    count = {k: 0 for k in state}
    seen = {k: 0 for k in state}
    boxes = set()
    pos = 0
    while any(s in (DC, DB) for s in state.values()):
        end = min(pos + block, n)
        active = {k for k, s in state.items() if s in (DC, DB)}
        maxlen = max(len(k) for k in active)
        for t in transactions[pos:end]:
            st = sorted(t)
            for r in range(1, min(maxlen, len(st)) + 1):
                for sub in combinations(st, r):
                    fs = frozenset(sub)
                    if fs in active:
                        count[fs] += 1
        for k in active:
            seen[k] += end - pos
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
        pos = end % n
    return {k: count[k] for k, s in state.items() if s == SB}, n


if __name__ == "__main__":
    tx = load_transactions()
    t0 = time.perf_counter()
    freq, n = dic(tx)
    rules = gen_rules(freq, n)
    el = time.perf_counter() - t0
    itemsets_df(freq, n).to_csv(TABLES / "dic_itemsets.csv", index=False)
    rules.to_csv(TABLES / "dic_rules.csv", index=False)
    record_time("DIC", "python", el, n)
    print(f"DIC: {len(freq)} frequent itemsets, {len(rules)} rules, {el:.3f}s")
