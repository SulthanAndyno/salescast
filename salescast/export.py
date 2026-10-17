"""Ekspor data dashboard → site/data.json (dibaca site/index.html di GitHub Pages).

python -m salescast.export
"""

import json

import numpy as np
import pandas as pd

from salescast.backtest import panel
from salescast.ct import CHAMPION, MODELS, REPORTS
from salescast.data import DATA, load, read_cursor
from salescast.features import SERIES

SITE = DATA.parent / "site"
HISTORY = 60  # hari aktual terakhir yang ditampilkan sebelum forecast


def export() -> dict:
    cursor = read_cursor()
    wide = panel(load("train", cursor)).iloc[-HISTORY:]
    f = pd.read_parquet(REPORTS / "forecast.parquet")

    def rows(df: pd.DataFrame) -> list:
        return np.round(df.reindex(columns=wide.columns).to_numpy().T, 1).tolist()

    data = {
        "cursor": str(cursor.date()),
        "champion": json.loads(CHAMPION.read_text()) if CHAMPION.exists() else None,
        "history_dates": wide.index.strftime("%Y-%m-%d").tolist(),
        "forecast_dates": sorted(f["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "series": [[int(s), str(fam)] for s, fam in wide.columns],
        "actual": rows(wide),
        **{m: rows(f.pivot(index="date", columns=SERIES, values=m)) for m in MODELS},
        "metrics": [json.loads(line) for line in (REPORTS / "metrics.jsonl").read_text().splitlines() if line],
    }
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    return data


if __name__ == "__main__":
    d = export()
    print(f"site/data.json: {len(d['series'])} series, {len(d['metrics'])} baris metrik, kursor {d['cursor']}")
