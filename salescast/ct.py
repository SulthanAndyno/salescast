"""Satu langkah continuous training: "minggu baru datang".

1. Kursor maju 7 hari → 7 hari aktual baru terungkap.
2. Forecast minggu lalu dinilai terhadap aktual itu → baris baru di reports/metrics.jsonl (data monitoring).
3. Retrain di data ≤ kursor, forecast 16 hari ke depan → reports/forecast.parquet.

Run pertama (belum ada forecast) hanya membuat forecast, kursor tidak maju.

python -m salescast.ct
"""

import json

import numpy as np
import pandas as pd

from salescast.backtest import HORIZON, close_holidays, moving_average, panel, rmsle
from salescast.data import DATA, END, advance, load, read_cursor
from salescast.features import SERIES
from salescast.model import context, lgbm

REPORTS = DATA.parent / "reports"
MODELS = ("lightgbm", "moving_average_28")


def score(forecast: pd.DataFrame, wide: pd.DataFrame, prev: pd.Timestamp, new: pd.Timestamp) -> dict:
    """RMSLE forecast lama di hari-hari yang baru terungkap (prev, new]."""
    f = forecast[(forecast["date"] > prev) & (forecast["date"] <= new)]
    actual = wide.stack(SERIES).rename("actual").rename_axis(["date", *SERIES])
    f = f.join(actual, on=["date", *SERIES], how="inner")
    row = {"cursor": str(new.date()), "days": int(f["date"].nunique())}
    row |= {m: round(rmsle(f["actual"], f[m]), 4) for m in MODELS}
    return row


def forecast(wide: pd.DataFrame, cursor: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(cursor + pd.Timedelta(days=1), periods=HORIZON)
    preds = {
        "lightgbm": lgbm(context(cursor))(wide, dates),
        "moving_average_28": moving_average(wide, dates),
    }
    long = {m: close_holidays(p).stack(SERIES) for m, p in preds.items()}
    return pd.DataFrame(long).rename_axis(["date", *SERIES]).reset_index()


def step() -> None:
    REPORTS.mkdir(exist_ok=True)
    fpath = REPORTS / "forecast.parquet"
    cursor = read_cursor()
    if fpath.exists():
        if cursor >= END:
            print(f"kursor sudah di akhir data ({END.date()}), simulasi selesai")
            return
        new = advance()
        row = score(pd.read_parquet(fpath), panel(load("train", new)), cursor, new)
        with open(REPORTS / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print("realisasi minggu ini:", row)
        cursor = new

    wide = panel(load("train", cursor))
    out = forecast(wide, cursor)
    out.to_parquet(fpath, index=False)
    print(
        f"forecast {out['date'].min().date()} … {out['date'].max().date()}, total lightgbm {np.round(out['lightgbm'].sum())}"
    )


if __name__ == "__main__":
    step()
