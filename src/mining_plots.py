import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import ROOT, TABLES

P = ROOT / "outputs" / "plots"
sns.set_theme(style="whitegrid")
rows = []


def save(name, inf):
    plt.tight_layout()
    plt.savefig(P / f"{name}.png", dpi=150)
    plt.close()
    rows.append((f"{name}.png", inf))


r = pd.read_csv(TABLES / "dic_rules.csv")
plt.figure(figsize=(9, 6))
sc = plt.scatter(r["support"], r["confidence"], c=r["lift"], cmap="viridis", s=40, alpha=0.8)
plt.colorbar(sc, label="lift")
for _, x in r.head(5).iterrows():
    plt.annotate(f"{x['antecedent']} -> {x['consequent']}", (x["support"], x["confidence"]), fontsize=7)
plt.xscale("log")
plt.title("DIC Association Rules: Support vs Confidence (colour = lift)")
plt.xlabel("support (log)")
plt.ylabel("confidence")
save("14_dic_rules_scatter", f"{len(r)} rules; {(r['lift'] > 1).mean() * 100:.1f}% have lift > 1; top lift {r['lift'].max():.2f} ({r.loc[0, 'antecedent']} -> {r.loc[0, 'consequent']}).")

d = pd.read_csv(TABLES / "dic_itemsets.csv").groupby("size").size()
a = pd.read_csv(TABLES / "aclose_closed_itemsets.csv").groupby("size").size()
c = pd.DataFrame({"DIC frequent": d, "A-Close closed": a}).fillna(0).astype(int)
c.plot(kind="bar", figsize=(8, 5))
plt.title("Frequent vs Closed Itemsets by Size")
plt.xlabel("itemset size")
plt.ylabel("count")
plt.xticks(rotation=0)
save("15_dic_vs_aclose_itemsets", f"A-Close keeps {c['A-Close closed'].sum()} of {c['DIC frequent'].sum()} itemsets ({(1 - c['A-Close closed'].sum() / c['DIC frequent'].sum()) * 100:.1f}% redundant removed) with no loss of support information.")

k = pd.read_csv(TABLES / "kmedoids_k_scores.csv")
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(k["k"], k["silhouette"], marker="o", color="#2a6fdb", label="silhouette")
ax1.set_xlabel("k")
ax1.set_ylabel("silhouette", color="#2a6fdb")
ax2 = ax1.twinx()
ax2.plot(k["k"], k["cost"], marker="s", color="#e8590c", label="cost")
ax2.set_ylabel("total Hamming cost", color="#e8590c")
plt.title("K-Medoids: Silhouette and Cost vs k")
bk = k.loc[k["silhouette"].idxmax()]
save("16_kmedoids_k_selection", f"Best k={int(bk['k'])} (silhouette {bk['silhouette']:.3f}); all other k score {k.drop(bk.name)['silhouette'].min():.2f}-{k.drop(bk.name)['silhouette'].max():.2f}, so clusters are weakly separated.")

pr = pd.read_csv(TABLES / "kmedoids_cluster_profiles.csv")
pr["label"] = pr["attribute"] + "=" + pr["top_value"]
pv = pr.pivot(index="attribute", columns="cluster", values="share")
lb = pr.pivot(index="attribute", columns="cluster", values="label")
plt.figure(figsize=(9, 5))
sns.heatmap(pv, annot=lb.values, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "share of cluster"})
plt.title("K-Medoids Cluster Profiles (most common value per attribute)")
mx = pr[pr["attribute"] == "Medal"]["share"]
save("17_kmedoids_cluster_profiles", f"Clusters split by Medal ({mx.min() * 100:.0f}%-{mx.max() * 100:.0f}% of each cluster share one medal type); other attributes are mixed.")

inf = pd.read_csv(TABLES / "plot_inferences.csv")
inf = pd.concat([inf[~inf["plot"].isin([x for x, _ in rows])], pd.DataFrame(rows, columns=["plot", "inference"])], ignore_index=True)
inf.to_csv(TABLES / "plot_inferences.csv", index=False)
