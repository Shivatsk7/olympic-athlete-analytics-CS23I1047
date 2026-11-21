# Olympic Athlete Performance Analytics

T. Shiva Kaushik (CS23I1047)

## Setup
- Python 3.12, OpenJDK 17
- python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Run
- bash run_all.sh

## Scripts (src/)
| Script | Purpose | Outputs |
|---|---|---|
| preprocess.py | Cleaning, selection, transformation | data/clean.csv, preprocess_summary.csv |
| plots.py | 12 descriptive plots + inferences | outputs/plots/01-12, plot_inferences.csv |
| dic.py | DIC frequent itemsets + rules | dic_itemsets.csv, dic_rules.csv |
| aclose.py | A-Close closed itemsets + rules | aclose_closed_itemsets.csv, aclose_rules.csv |
| kmedoids.py | K-Medoids (Hamming), k=2..8 | kmedoids_k_scores.csv, kmedoids_clusters.csv, kmedoids_cluster_profiles.csv |
| mining_plots.py | ARM rule scatter, DIC vs A-Close, K-Medoids k-selection and profiles | outputs/plots/14-17 |
| scale.py | Replicated datasets x1-x200 | data/scaled_x*.csv |
| dic_spark.py | DIC in PySpark vs Python | spark_dic_runs.csv, spark_dic_verify.csv |
| kmedoids_spark.py | K-Medoids in PySpark vs Python | spark_kmedoids_runs.csv, spark_kmedoids_verify.csv |
| spark_compare.py | Speedup table + plot | spark_vs_python.csv, 13_spark_vs_python.png |

## Parameters
- min_support 0.01, min_confidence 0.5
- Transactions: {Sport, Gender, NOC, Medal, Era}
- K-Medoids: Hamming distance, sample 5000, seed 42
- Counts are medal awards per athlete; team events count every member
