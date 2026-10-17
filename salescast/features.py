"""Fitur untuk strategi direct: semua fitur penjualan memakai data ≥ 16 hari sebelum tanggal target,
jadi satu model bisa memprediksi H+1..H+16 dari cutoff yang sama tanpa melihat masa depan."""

import pandas as pd

from salescast.backtest import HORIZON

LAGS = [16, 21, 28, 35, 42, 364]  # 21/28/35/42 = hari yang sama 3–6 minggu lalu, 364 = tahun lalu
HOLIDAY_TYPES = ["Holiday", "Transfer", "Additional", "Bridge"]
SERIES = ["store_nbr", "family"]


def _long(wide: pd.DataFrame) -> pd.Series:
    return wide.stack(SERIES)


def gaps(wide: pd.DataFrame) -> pd.DataFrame:
    """True di (tanggal, kategori) yang total nasionalnya < 1% dari rekor sebelumnya: data hilang
    (mis. PRODUCE awal 2015 tercatat ~0 berbulan-bulan) atau toko tutup massal (25 Des, 1 Jan).
    Bukan "laku 0", jadi tidak boleh dipakai sebagai fitur maupun target."""
    fam = wide.T.groupby(level="family").sum().T
    flag = fam < 0.01 * fam.cummax().shift(1)
    return flag.reindex(columns=wide.columns, level="family")


def sales_features(wide: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """wide boleh berisi NaN untuk tanggal target di masa depan; shift ≥ 16 tidak pernah menyentuhnya."""
    wide = wide.mask(gaps(wide))
    out = {f"lag_{k}": wide.shift(k) for k in LAGS}
    base = wide.shift(HORIZON)
    for w in (7, 14, 28, 56):
        out[f"mean_{w}"] = base.rolling(w, min_periods=1).mean()
    out["std_28"] = base.rolling(28, min_periods=2).std()
    out["dow_mean_4"] = sum(wide.shift(k) for k in (21, 28, 35, 42)) / 4
    return pd.concat({k: _long(v.loc[start:]) for k, v in out.items()}, axis=1)


def holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Flag per (date, store_nbr): libur nasional, libur regional/lokal di kota/provinsi toko, event, hari kerja pengganti."""
    h = holidays[~holidays["transferred"]]
    nat = h[h["locale"] == "National"]
    by_date = pd.DataFrame(
        {
            "holiday": nat[nat["type"].isin(HOLIDAY_TYPES)].groupby("date").size() > 0,
            "event": nat[nat["type"] == "Event"].groupby("date").size() > 0,
            "workday": nat[nat["type"] == "Work Day"].groupby("date").size() > 0,
        }
    ).fillna(False)

    loc = h[h["locale"] != "National"]
    local = pd.concat(
        [
            loc[loc["locale"] == "Regional"].merge(stores, left_on="locale_name", right_on="state"),
            loc[loc["locale"] == "Local"].merge(stores, left_on="locale_name", right_on="city"),
        ]
    )[["date", "store_nbr"]].drop_duplicates()
    local["local_holiday"] = True

    full = stores[["store_nbr"]].merge(by_date.reset_index(names="date"), how="cross")
    return full.merge(local, on=["date", "store_nbr"], how="outer").fillna(False).set_index(["date", "store_nbr"])


def build(wide: pd.DataFrame, ctx: dict, start: pd.Timestamp) -> pd.DataFrame:
    """Satu baris per (date, store_nbr, family) dari `start` sampai akhir `wide`; kolom `sales` = target."""
    X = sales_features(wide, start)
    X["sales"] = _long(wide.loc[start:])
    X["gap"] = _long(gaps(wide).loc[start:])
    X["onpromotion"] = _long(ctx["promo"].reindex(wide.index).loc[start:])
    X = X.rename_axis(["date", *SERIES]).reset_index()

    d = X["date"].dt
    X["dayofweek"], X["day"], X["month"] = d.dayofweek, d.day, d.month
    X["payday"] = (d.day == 15) | d.is_month_end  # gaji sektor publik Ekuador: tgl 15 & akhir bulan

    oil = ctx["oil"].set_index("date")["dcoilwtico"]
    oil = oil.reindex(pd.date_range(oil.index.min(), wide.index.max())).ffill().shift(HORIZON)
    X["oil"] = X["date"].map(oil)

    X = X.merge(ctx["holidays"], left_on=["date", "store_nbr"], right_index=True, how="left")
    X = X.merge(ctx["stores"][["store_nbr", "type", "cluster"]], on="store_nbr", how="left")
    for c in ("holiday", "event", "workday", "local_holiday"):
        X[c] = X[c].fillna(False).astype(bool)
    for c in ("store_nbr", "family", "type", "cluster"):
        X[c] = X[c].astype("category")
    return X
