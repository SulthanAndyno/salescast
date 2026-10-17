"""Backtest walk-forward: potong data di beberapa cutoff, prediksi 16 hari, hitung RMSLE.

python -m salescast.backtest
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from salescast.data import load, read_cursor

HORIZON = 16
Forecaster = Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]


def panel(train: pd.DataFrame) -> pd.DataFrame:
    """Tabel lebar: baris = tanggal, kolom = (store_nbr, family). Tanggal yang hilang (25 Des, toko tutup) diisi 0."""
    wide = train.pivot(index="date", columns=["store_nbr", "family"], values="sales")
    return wide.reindex(pd.date_range(wide.index.min(), wide.index.max()), fill_value=0.0)


def rmsle(actual, pred) -> float:
    actual, pred = np.asarray(actual, dtype=float), np.clip(np.asarray(pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def seasonal_naive(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Hari yang sama minggu lalu."""
    last_week = hist.iloc[-7:]
    rows = [last_week.iloc[(i % 7)].to_numpy() for i in range(len(dates))]
    return pd.DataFrame(rows, index=dates, columns=hist.columns)


def moving_average(hist: pd.DataFrame, dates: pd.DatetimeIndex, window: int = 28) -> pd.DataFrame:
    mean = hist.iloc[-window:].mean().to_numpy()
    return pd.DataFrame(np.tile(mean, (len(dates), 1)), index=dates, columns=hist.columns)


def cutoffs(cursor: pd.Timestamp, n: int = 6) -> list[pd.Timestamp]:
    """n cutoff mingguan; cutoff terakhir = cursor - 16 hari supaya aktualnya masih ≤ kursor."""
    last = cursor - pd.Timedelta(days=HORIZON)
    return [last - pd.Timedelta(weeks=k) for k in range(n - 1, -1, -1)]


def close_holidays(pred: pd.DataFrame) -> pd.DataFrame:
    """25 Des semua toko tutup, 1 Jan hampir semua tutup: dua-duanya sudah pasti dari kalender, jadi prediksi 0."""
    # ponytail: aturan tanggal tetap; ganti fitur "toko tutup" per toko kalau nanti ada tanggal tutup lain
    closed = (pred.index.month == 12) & (pred.index.day == 25) | (pred.index.month == 1) & (pred.index.day == 1)
    pred = pred.copy()
    pred.loc[closed] = 0.0
    return pred


def backtest(forecaster: Forecaster, wide: pd.DataFrame, cuts: list[pd.Timestamp]) -> list[float]:
    scores = []
    for c in cuts:
        dates = pd.date_range(c + pd.Timedelta(days=1), periods=HORIZON)
        pred = close_holidays(forecaster(wide.loc[:c], dates))
        scores.append(rmsle(wide.loc[dates].to_numpy(), pred.to_numpy()))
    return scores


if __name__ == "__main__":
    cursor = read_cursor()
    wide = panel(load("train", cursor))
    cuts = cutoffs(cursor)
    print(f"kursor {cursor.date()}, cutoff {cuts[0].date()} … {cuts[-1].date()}")
    for name, f in {"seasonal_naive": seasonal_naive, "moving_average_28": moving_average}.items():
        s = backtest(f, wide, cuts)
        print(f"{name:18} RMSLE rata-rata {np.mean(s):.4f}  per cutoff {[round(x, 3) for x in s]}")
