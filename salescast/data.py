"""Download data Favorita dari Kaggle + kursor waktu untuk simulasi "data baru tiap minggu".

python -m salescast.data download   # butuh KAGGLE_USERNAME/KAGGLE_KEY & rules kompetisi sudah di-accept
python -m salescast.data cursor     # tampilkan kursor
python -m salescast.data advance    # maju 7 hari
"""

import sys
from pathlib import Path

import pandas as pd

COMPETITION = "store-sales-time-series-forecasting"
DATA = Path(__file__).resolve().parent.parent / "data"
CURSOR = DATA / "cursor.txt"
FILES = ["train", "stores", "oil", "holidays_events", "transactions"]  # test.csv tidak berlabel, tidak dipakai
END = pd.Timestamp("2017-08-15")  # tanggal terakhir di train.csv


def download(dest: Path = DATA) -> None:
    import kagglehub

    src = Path(kagglehub.competition_download(COMPETITION))
    dest.mkdir(exist_ok=True)
    for name in FILES:
        df = pd.read_csv(src / f"{name}.csv", parse_dates=["date"] if name != "stores" else None)
        df.to_parquet(dest / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} baris")


def read_cursor(path: Path = CURSOR) -> pd.Timestamp:
    return pd.Timestamp(path.read_text().strip())


def advance(days: int = 7, path: Path = CURSOR) -> pd.Timestamp:
    new = min(read_cursor(path) + pd.Timedelta(days=days), END)
    path.write_text(f"{new.date()}\n")
    return new


def load(name: str, cursor: pd.Timestamp | None = None, data: Path = DATA) -> pd.DataFrame:
    """Baca tabel; kalau ada kursor, buang baris setelah kursor (data "masa depan" belum diketahui)."""
    df = pd.read_parquet(data / f"{name}.parquet")
    if cursor is not None and "date" in df:
        df = df[df["date"] <= cursor]
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cursor"
    if cmd == "download":
        download()
    elif cmd == "advance":
        print(advance())
    elif cmd == "cursor":
        print(read_cursor().date())
    else:
        sys.exit(f"perintah tidak dikenal: {cmd}")
