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


def fit(ctx: dict, wide: pd.DataFrame, cutoff: pd.Timestamp) -> lgb.Booster:
    """Latih di target ≤ cutoff (TRAIN_DAYS terakhir); data setelah cutoff tidak pernah disentuh."""
    X = build(wide.loc[:cutoff], ctx, start=cutoff - pd.Timedelta(days=TRAIN_DAYS))
    train = X[~X["gap"] & ~X["date"].between(*QUAKE)]
    cols = features_of(X)
    return lgb.train(PARAMS, lgb.Dataset(train[cols], np.log1p(train["sales"])), num_boost_round=ROUNDS)


def predict(model: lgb.Booster, ctx: dict, wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast `dates` memakai data sebelum dates[0]; model boleh dilatih di cutoff yang lebih lama."""
    hist = wide.loc[: dates[0] - pd.Timedelta(days=1)]
    X = build(hist.reindex(pd.date_range(hist.index[0], dates[-1])), ctx, start=dates[0])
    pred = X[["date", *SERIES]].assign(sales=np.expm1(model.predict(X[model.feature_name()])).clip(0))
    pred = pred.set_index(["date", *SERIES])["sales"].unstack(SERIES).reindex(columns=hist.columns)
    dead = hist.iloc[-56:].sum() == 0  # tidak terjual 8 minggu → prediksi 0
    pred.loc[:, dead[dead].index] = 0.0
    return pred


def lgbm(ctx: dict, importance: list | None = None) -> Forecaster:
    def forecast(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        model = fit(ctx, hist, hist.index[-1])
        if importance is not None:
            importance.append(pd.Series(model.feature_importance("gain"), index=model.feature_name()))
        return predict(model, ctx, hist, dates)

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
