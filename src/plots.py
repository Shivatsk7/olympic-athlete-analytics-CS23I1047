import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from preprocess import ATHLETE_KEY

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "clean.csv")
P = ROOT / "outputs" / "plots"
T = ROOT / "outputs" / "tables"
P.mkdir(parents=True, exist_ok=True)
T.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
MEDALS = ["Gold", "Silver", "Bronze"]
MC = {"Gold": "#D4AF37", "Silver": "#A8A9AD", "Bronze": "#CD7F32"}
ERAS = sorted(df["Era"].unique(), key=lambda e: int(e.split("_")[-1].split("-")[0]))
rows = []


def save(name, inf):
    plt.tight_layout()
    plt.savefig(P / f"{name}.png", dpi=150)
    plt.close()
    rows.append((f"{name}.png", inf))


ed = df.groupby("Edition").size()
plt.figure(figsize=(10, 5))
plt.plot(ed.index, ed.values, marker="o")
plt.title("Medal Awards per Edition")
plt.xlabel("Edition")
plt.ylabel("Medal Awards")
save("01_medals_per_edition", f"Medal awards grew {ed.max() / ed.iloc[0]:.1f}x from {ed.iloc[0]} in {ed.index[0]} to a peak of {ed.max()} in {ed.idxmax()}.")

me = pd.crosstab(df["Era"], df["Medal"]).reindex(index=ERAS, columns=MEDALS)
me.plot(kind="bar", stacked=True, color=[MC[m] for m in MEDALS], figsize=(10, 5))
plt.title("Medal Type by Era")
plt.ylabel("Medal Awards")
plt.xticks(rotation=20)
gs = me["Gold"] / me.sum(axis=1) * 100
save("02_medal_type_by_era", f"Gold share stays near one third in every era ({gs.min():.1f}%-{gs.max():.1f}%); {me.sum(axis=1).idxmax()} has the most medal awards ({me.sum(axis=1).max()}).")

tn = df["NOC"].value_counts().head(15)
plt.figure(figsize=(9, 6))
plt.barh(tn.index[::-1], tn.values[::-1])
plt.title("Top 15 NOCs by Medal Awards")
plt.xlabel("Medal Awards")
ev = df.drop_duplicates(["Edition", "Discipline", "Event", "Event_gender", "Medal", "NOC"])["NOC"].value_counts()
save("03_top15_nocs", f"{tn.index[0]} leads with {tn.iloc[0]} medal awards, {tn.iloc[0] / tn.iloc[1]:.1f}x second-placed {tn.index[1]} ({tn.iloc[1]}); counting each team once: {tn.index[0]} {ev[tn.index[0]]}, {tn.index[1]} {ev[tn.index[1]]}.")

g = df["Gender"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(g.values, labels=g.index, autopct="%1.1f%%", startangle=90)
plt.title("Gender Share of Medal Awards")
save("04_gender_share", f"{g.index[0]} won {g.iloc[0] / g.sum() * 100:.1f}% of all medal awards vs {g.index[1]} {g.iloc[1] / g.sum() * 100:.1f}%.")

ge = pd.crosstab(df["Edition"], df["Gender"]).fillna(0)
plt.figure(figsize=(10, 5))
plt.stackplot(ge.index, [ge[c] for c in ge.columns], labels=ge.columns, alpha=0.8)
plt.legend(loc="upper left")
plt.title("Men vs Women Medal Awards per Edition")
plt.xlabel("Edition")
plt.ylabel("Medal Awards")
wf = ge["Women"] / ge.sum(axis=1) * 100
first_w = ge.index[ge["Women"] > 0][0]
save("05_gender_per_edition", f"Women's share grew from 0% in {ge.index[0]} (first medal awards {first_w}) to {wf.iloc[-1]:.1f}% in {ge.index[-1]}.")

t10n = df["NOC"].value_counts().head(10).index
t10s = df["Sport"].value_counts().head(10).index
hm = pd.crosstab(df["NOC"], df["Sport"]).loc[t10n, t10s]
plt.figure(figsize=(11, 7))
sns.heatmap(hm, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Top 10 NOCs x Top 10 Sports")
mx = hm.stack().idxmax()
save("06_noc_sport_heatmap", f"Strongest cell is {mx[0]} in {mx[1]} ({hm.loc[mx]} medal awards), showing country-sport specialisation.")

pa = df.groupby(ATHLETE_KEY).size().rename("n").reset_index()
plt.figure(figsize=(10, 5))
plt.hist(pa["n"], bins=range(1, pa["n"].max() + 2), log=True)
plt.title("Medal Awards per Athlete")
plt.xlabel("Medal Awards")
plt.ylabel("Athletes (log)")
save("07_medals_per_athlete_hist", f"{(pa['n'] == 1).mean() * 100:.1f}% of {len(pa)} athletes won a single medal; max is {pa['n'].max()} ({pa.loc[pa['n'].idxmax(), 'Athlete']}).")

ne = df.groupby(["Era", "Edition", "NOC"]).size().rename("n").reset_index()
plt.figure(figsize=(10, 5))
sns.boxplot(data=ne, x="Era", y="n", order=ERAS)
plt.yscale("log")
plt.title("Medal Awards per NOC per Edition, by Era")
plt.xticks(rotation=20)
md = ne.groupby("Era")["n"].median().reindex(ERAS)
save("08_noc_edition_box_era", f"Median medal awards per NOC per edition ranges {md.min():.0f}-{md.max():.0f} across eras while outliers reach {ne['n'].max()}, a heavily skewed distribution.")

plt.figure(figsize=(8, 5))
sns.violinplot(data=pa, x="Gender", y="n", cut=0)
plt.title("Medal Awards per Athlete by Gender")
mg = pa.groupby("Gender")["n"].agg(["mean", "max"])
save("09_athlete_violin_gender", f"Mean medal awards per athlete: Men {mg.loc['Men', 'mean']:.2f} (max {mg.loc['Men', 'max']}), Women {mg.loc['Women', 'mean']:.2f} (max {mg.loc['Women', 'max']}).")

gb = pd.crosstab(df["NOC"], df["Medal"]).loc[t10n, MEDALS]
gb.plot(kind="bar", color=[MC[m] for m in MEDALS], figsize=(11, 5))
plt.title("Gold/Silver/Bronze for Top 10 NOCs")
plt.ylabel("Medal Awards")
gr = gb["Gold"] / gb.sum(axis=1)
save("10_top10_noc_medal_types", f"{gr.idxmax()} has the highest gold ratio among top 10 ({gr.max() * 100:.1f}%), {gr.idxmin()} the lowest ({gr.min() * 100:.1f}%).")

bb = df.groupby("NOC").agg(eds=("Edition", "nunique"), tot=("Medal", "size"), gold=("Medal", lambda s: (s == "Gold").sum()))
plt.figure(figsize=(10, 6))
plt.scatter(bb["eds"], bb["tot"], s=bb["gold"] / bb["gold"].max() * 1500 + 10, alpha=0.5)
for n in bb["tot"].nlargest(8).index:
    plt.annotate(n, (bb.loc[n, "eds"], bb.loc[n, "tot"]))
plt.yscale("log")
plt.title("Editions with Medals vs Total Medal Awards per NOC (size = golds)")
plt.xlabel("Editions with medal awards")
plt.ylabel("Total medal awards (log)")
r = np.corrcoef(bb["eds"], np.log(bb["tot"]))[0, 1]
save("11_bubble_editions_vs_medals", f"Editions with medal awards correlate with log total medal awards (r={r:.2f}) across {len(bb)} NOCs.")

td = df["Discipline"].value_counts().head(15)
plt.figure(figsize=(9, 6))
sns.countplot(data=df[df["Discipline"].isin(td.index)], y="Discipline", order=td.index)
plt.title("Top 15 Disciplines by Medal Awards")
save("12_top15_disciplines", f"{td.index[0]} ({td.iloc[0]}) and {td.index[1]} ({td.iloc[1]}) dominate, together {(td.iloc[0] + td.iloc[1]) / len(df) * 100:.1f}% of all medal awards.")

pd.DataFrame(rows, columns=["plot", "inference"]).to_csv(T / "plot_inferences.csv", index=False)
