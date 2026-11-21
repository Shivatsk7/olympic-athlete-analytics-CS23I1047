import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import ROOT, TABLES

P = ROOT / "outputs" / "plots"
sns.set_theme(style="whitegrid")

runs = pd.concat([pd.read_csv(TABLES / f"spark_{a}_runs.csv") for a in ["dic", "kmedoids"]], ignore_index=True)
runs["speedup"] = (runs["python_s"] / runs["spark_s"]).round(3)
runs["repeats"] = runs["repeats"].astype(int)
runs.to_csv(TABLES / "spark_vs_python.csv", index=False)
pd.concat([pd.read_csv(TABLES / f"spark_{a}_verify.csv") for a in ["dic", "kmedoids"]]).to_csv(TABLES / "spark_verification.csv", index=False)
pd.concat([pd.read_csv(TABLES / f"spark_session_{a}.csv") for a in ["dic", "kmedoids"]]).to_csv(TABLES / "spark_session_start.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (algo, g) in zip(axes, runs.groupby("algorithm", sort=False)):
    ax.plot(g["size"], g["python_s"], marker="o", color="#2a6fdb", label="Python")
    ax.plot(g["size"], g["spark_s"], marker="s", color="#e8590c", label="PySpark local[*]")
    for s, sp, y in zip(g["size"], g["speedup"], g[["python_s", "spark_s"]].max(axis=1)):
        ax.annotate(f"{sp:.2f}x", (s, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{algo}: runtime vs size")
    ax.set_xlabel("rows")
    ax.set_ylabel("seconds (log)")
    ax.legend()
fig.suptitle("Python vs PySpark runtime (labels = speedup, Python/Spark)")
plt.tight_layout()
plt.savefig(P / "13_spark_vs_python.png", dpi=150)
