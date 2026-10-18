"""Submission Kaggle: latih di seluruh train.csv, forecast 16 hari test.csv (16–31 Agustus 2017).

python -m salescast.submit   # tulis submission.csv; terpisah dari CT (kursor dan champion tidak berubah)
"""

from pathlib import Path

import pandas as pd

from salescast.backtest import close_holidays, panel
from salescast.data import COMPETITION, DATA, END, load
from salescast.features import SERIES
from salescast.model import context, fit, predict

OUT = DATA.parent / "submission.csv"


def main() -> None:
    import kagglehub

    test = pd.read_csv(Path(kagglehub.competition_download(COMPETITION)) / "test.csv", parse_dates=["date"])
    train = load("train")
    ctx = context(END)
    # promo periode test hanya ada di test.csv
    both = pd.concat([train, test]).rename(columns={"sales": "_", "onpromotion": "sales"})
    ctx["promo"] = panel(both)
    wide = panel(train)
    dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    pred = close_holidays(predict(fit(ctx, wide, END), ctx, wide, dates))
    sales = pred.stack(SERIES).rename("sales").rename_axis(["date", *SERIES]).reset_index()
    sub = test.merge(sales, on=["date", *SERIES], how="left")
    assert len(sub) == len(test) and sub["sales"].notna().all(), "ada id test tanpa prediksi"
    sub[["id", "sales"]].to_csv(OUT, index=False)
    print(f"{OUT}: {len(sub):,} baris, total {sub['sales'].sum():,.0f}")


if __name__ == "__main__":
    main()
