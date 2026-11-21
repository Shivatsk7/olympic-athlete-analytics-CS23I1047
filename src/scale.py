import sys
import pandas as pd
from common import ROOT, load_df

SCALES = [1, 10, 50]


def scaled_path(n):
    return ROOT / "data" / f"scaled_x{n}.csv"


if __name__ == "__main__":
    df = load_df()
    for n in [int(a) for a in sys.argv[1:]] or SCALES:
        out = pd.concat([df] * n, ignore_index=True)
        out.to_csv(scaled_path(n), index=False)
        print(n, len(out))
