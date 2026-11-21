import time
from itertools import combinations
import pandas as pd
from common import TABLES, MIN_SUPPORT, MIN_CONF, load_transactions, itemsets_df, rule_row, rules_df, record_time


def tidsets(transactions):
    tid = {}
    for i, t in enumerate(transactions):
        for it in t:
            tid[it] = tid.get(it, 0) | (1 << i)
    return tid


def aclose(transactions, min_support=MIN_SUPPORT):
    n = len(transactions)
    minc = min_support * n
    tid = tidsets(transactions)
    full = (1 << n) - 1
    gens = {}
    level = {}
    for it, b in tid.items():
        c = b.bit_count()
        if c >= minc:
            level[frozenset([it])] = (b, c)
    gens.update(level)
    while level:
        keys = sorted(level, key=lambda k: sorted(k))
        nxt = {}
        for a, b in combinations(keys, 2):
            c = a | b
            if len(c) != len(a) + 1 or c in nxt:
                continue
            subs = [c - {z} for z in c]
            if not all(s in level for s in subs):
                continue
            bits = level[a][0] & level[b][0]
            cnt = bits.bit_count()
            if cnt < minc or any(level[s][1] == cnt for s in subs):
                continue
            nxt[c] = (bits, cnt)
        gens.update(nxt)
        level = nxt
    closed = {}
    gen_closure = {}
    for g, (bits, cnt) in gens.items():
        cl = frozenset(it for it, b in tid.items() if bits & b == bits)
        closed[cl] = cnt
        gen_closure[g] = cl
    return closed, gen_closure, {g: v[1] for g, v in gens.items()}, n


def support_of(x, closed):
    return max(v for c, v in closed.items() if x <= c)


def aclose_rules(closed, gen_closure, gen_sup, n):
    rows = []
    for g, cl in gen_closure.items():
        sg = gen_sup[g]
        for c, sc in closed.items():
            if cl <= c and c != g:
                conf = sc / sg
                if conf >= MIN_CONF:
                    cons = c - g
                    rows.append(rule_row(g, cons, sc / n, conf, support_of(cons, closed) / n))
    return rules_df(rows).drop_duplicates(["antecedent", "consequent"]).reset_index(drop=True)


if __name__ == "__main__":
    tx = load_transactions()
    t0 = time.perf_counter()
    closed, gc, gs, n = aclose(tx)
    rules = aclose_rules(closed, gc, gs, n)
    el = time.perf_counter() - t0
    itemsets_df(closed, n).to_csv(TABLES / "aclose_closed_itemsets.csv", index=False)
    rules.to_csv(TABLES / "aclose_rules.csv", index=False)
    record_time("A-Close", "python", el, n)
    dic_n = len(pd.read_csv(TABLES / "dic_itemsets.csv")) if (TABLES / "dic_itemsets.csv").exists() else None
    print(f"A-Close: {len(gs)} generators, {len(closed)} closed itemsets vs DIC {dic_n} frequent itemsets, {len(rules)} rules, {el:.3f}s")
