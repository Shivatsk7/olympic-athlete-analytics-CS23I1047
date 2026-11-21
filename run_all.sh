#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
export PYSPARK_PYTHON="../.venv/bin/python"
PY=../.venv/bin/python
$PY preprocess.py
$PY plots.py
$PY dic.py
$PY aclose.py
$PY kmedoids.py
$PY mining_plots.py
$PY scale.py 1 10 50 200
$PY dic_spark.py 1 10 50 200
$PY kmedoids_spark.py 2000 5000 10000 20000 40000
$PY spark_compare.py
