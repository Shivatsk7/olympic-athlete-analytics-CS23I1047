import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "olympics.csv"
CLEAN = ROOT / "data" / "clean.csv"
SUMMARY = ROOT / "outputs" / "tables" / "preprocess_summary.csv"

ERA_BINS = [1895, 1912, 1936, 1968, 1988, 2008]
ERA_LABELS = ["Early_1896-1912", "Interwar_1920-1936", "Postwar_1948-1968", "ColdWar_1972-1988", "Modern_1992-2008"]
MEDAL_POINTS = {"Gold": 3, "Silver": 2, "Bronze": 1}
ATHLETE_KEY = ["Athlete", "NOC", "Gender"]


def main():
    df = pd.read_csv(RAW, skiprows=4)
    stats = [("raw_rows", len(df)), ("raw_columns", df.shape[1]), ("missing_cells", int(df.isna().sum().sum()))]

    df = df.dropna()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
    df["Athlete"] = df["Athlete"].str.title()
    df["Edition"] = df["Edition"].astype(int)

    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    stats.append(("duplicates_removed", dups))

    bad = (df["Gender"].eq("Men") & df["Event_gender"].eq("W")) | (df["Gender"].eq("Women") & df["Event_gender"].eq("M"))
    stats.append(("gender_mismatch_removed", int(bad.sum())))
    df = df[~bad]

    df = df[df["Medal"].isin(MEDAL_POINTS)]
    df["Era"] = pd.cut(df["Edition"], bins=ERA_BINS, labels=ERA_LABELS).astype(str)
    df["Decade"] = (df["Edition"] // 10 * 10).astype(int)
    df["Medal_points"] = df["Medal"].map(MEDAL_POINTS)

    cols = ["Edition", "Era", "Decade", "City", "Sport", "Discipline", "Event", "Event_gender", "Athlete", "NOC", "Gender", "Medal", "Medal_points"]
    df = df[cols].sort_values(["Edition", "Sport", "Event", "Medal_points"], ascending=[True, True, True, False]).reset_index(drop=True)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)

    stats += [
        ("clean_rows", len(df)),
        ("clean_columns", df.shape[1]),
        ("editions", df["Edition"].nunique()),
        ("edition_range", f"{df['Edition'].min()}-{df['Edition'].max()}"),
        ("sports", df["Sport"].nunique()),
        ("disciplines", df["Discipline"].nunique()),
        ("events", df["Event"].nunique()),
        ("nocs", df["NOC"].nunique()),
        ("unique_athletes", len(df[ATHLETE_KEY].drop_duplicates())),
        ("men_rows", int(df["Gender"].eq("Men").sum())),
        ("women_rows", int(df["Gender"].eq("Women").sum())),
        ("gold", int(df["Medal"].eq("Gold").sum())),
        ("silver", int(df["Medal"].eq("Silver").sum())),
        ("bronze", int(df["Medal"].eq("Bronze").sum())),
    ]
    stats += [(f"era_{e}", int(n)) for e, n in df["Era"].value_counts().reindex(ERA_LABELS).items()]
    pd.DataFrame(stats, columns=["metric", "value"]).to_csv(SUMMARY, index=False)
    print(pd.DataFrame(stats, columns=["metric", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
