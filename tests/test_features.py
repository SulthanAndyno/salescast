import numpy as np
import pandas as pd

from salescast.backtest import HORIZON, panel
from salescast.features import gaps, holiday_flags, sales_features


def random_wide(days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=days)
    rows = [(d, s, f, rng.poisson(5)) for d in dates for s in (1, 2) for f in ("A", "B")]
    return panel(pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales"]))


def test_anti_leakage_fitur_tidak_berubah_kalau_masa_depan_diubah():
    wide = random_wide()
    t = wide.index[450]
    changed = wide.copy()
    changed.loc[changed.index > t] = 9999.0

    start = wide.index[400]
    a = sales_features(wide, start).loc[: t + pd.Timedelta(days=HORIZON)]
    b = sales_features(changed, start).loc[: t + pd.Timedelta(days=HORIZON)]
    pd.testing.assert_frame_equal(a, b)
    # dan memang ada yang berubah sesudahnya, jadi test di atas tidak lolos karena kebetulan
    assert not sales_features(wide, start).equals(sales_features(changed, start))


def test_holiday_flags_nasional_dan_lokal():
    stores = pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Cuenca"], "state": ["Pichincha", "Azuay"]})
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-04-12", "2017-05-01"]),
            "type": ["Holiday", "Holiday", "Holiday"],
            "locale": ["National", "Local", "National"],
            "locale_name": ["Ecuador", "Cuenca", "Ecuador"],
            "transferred": [False, False, True],
        }
    )
    f = holiday_flags(holidays, stores)
    assert f.loc[("2017-01-01", 1), "holiday"] and f.loc[("2017-01-01", 2), "holiday"]
    assert f.loc[("2017-04-12", 2), "local_holiday"] and ("2017-04-12", 1) not in f.index  # baris tidak ada = False
    assert ("2017-05-01", 1) not in f.index  # libur yang dipindah (transferred) tidak dihitung


def test_gaps_menandai_kategori_yang_totalnya_anjlok():
    dates = pd.date_range("2015-01-01", periods=6)
    sales = {"A": [100, 120, 0, 1, 110, 0], "B": [0, 0, 0, 0, 0, 0]}  # B tidak pernah laku: bukan gap
    rows = [(d, s, f, sales[f][i] / 2) for i, d in enumerate(dates) for s in (1, 2) for f in "AB"]
    g = gaps(panel(pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales"])))
    assert g[(1, "A")].tolist() == [False, False, True, True, False, True]  # 1 < 1% dari 120
    assert not g[(2, "B")].any()
