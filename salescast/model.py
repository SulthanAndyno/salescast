"""LightGBM global (semua series satu model), target log1p(sales) → L2 loss = optimasi RMSLE.

python -m salescast.model   # backtest LightGBM vs baseline
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from salescast.backtest import Forecaster, backtest, cutoffs, moving_average, panel, seasonal_naive
from salescast.data import load, read_cursor
from salescast.features import SERIES, build, holiday_flags

TRAIN_DAYS = 730
ROUNDS = 600
QUAKE = (pd.Timestamp("2016-04-16"), pd.Timestamp("2016-05-15"))  # gempa Ekuador: lonjakan anomali, tidak dipelajari
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
}


def context(cursor: pd.Timestamp) -> dict:
    stores = load("stores")
    # onpromotion dibaca tanpa kursor: jadwal promo sudah diketahui sebelum harinya (test.csv Kaggle juga memberinya)
    promo = panel(load("train").rename(columns={"sales": "_", "onpromotion": "sales"}))
    return {
        "stores": stores,
        "promo": promo,
        "oil": load("oil", cursor),
        "holidays": holiday_flags(load("holidays_events"), stores),
    }


def features_of(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("date", "sales", "gap")]


def lgbm(ctx: dict, importance: list | None = None) -> Forecaster:
    def forecast(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        cutoff = hist.index[-1]
        wide = hist.reindex(pd.date_range(hist.index[0], dates[-1]))
        X = build(wide, ctx, start=cutoff - pd.Timedelta(days=TRAIN_DAYS))
        d = X["date"]
        train = X[(d <= cutoff) & ~X["gap"] & ~d.between(*QUAKE)]
        test = X[d.isin(dates)]

        cols = features_of(X)
        model = lgb.train(PARAMS, lgb.Dataset(train[cols], np.log1p(train["sales"])), num_boost_round=ROUNDS)
        if importance is not None:
            importance.append(pd.Series(model.feature_importance("gain"), index=cols))

        pred = test[["date", *SERIES]].assign(sales=np.expm1(model.predict(test[cols])).clip(0))
        pred = pred.set_index(["date", *SERIES])["sales"].unstack(SERIES).reindex(columns=hist.columns)
        dead = hist.iloc[-56:].sum() == 0  # tidak terjual 8 minggu → prediksi 0
        pred.loc[:, dead[dead].index] = 0.0
        return pred

    return forecast


if __name__ == "__main__":
    cursor = read_cursor()
    wide = panel(load("train", cursor))
    cuts = cutoffs(cursor)
    importance = []
    results = {
        "seasonal_naive": backtest(seasonal_naive, wide, cuts),
        "moving_average_28": backtest(moving_average, wide, cuts),
        "lightgbm": backtest(lgbm(context(cursor), importance), wide, cuts),
    }
    print(f"kursor {cursor.date()}, cutoff {cuts[0].date()} … {cuts[-1].date()}")
    for name, s in results.items():
        print(f"{name:18} RMSLE rata-rata {np.mean(s):.4f}  per cutoff {[round(x, 3) for x in s]}")
    gain = pd.concat(importance, axis=1).mean(axis=1)
    print("fitur terpenting:", (gain / gain.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
