import numpy as np
import pandas as pd

from salescast.backtest import HORIZON, backtest, close_holidays, cutoffs, moving_average, panel, rmsle, seasonal_naive


def weekly_panel(days=70):
    dates = pd.date_range("2017-01-01", periods=days)
    rows = [{"date": d, "store_nbr": s, "family": "A", "sales": float(d.dayofweek * s)} for d in dates for s in (1, 2)]
    return pd.DataFrame(rows)


def test_backtest_tidak_pernah_memberi_data_setelah_cutoff():
    wide = panel(weekly_panel())
    cuts = cutoffs(wide.index.max(), n=3)
    seen = []

    def spy(hist, dates):
        seen.append((hist.index.max(), dates))
        return moving_average(hist, dates)

    backtest(spy, wide, cuts)
    for c, (hist_max, dates) in zip(cuts, seen):
        assert hist_max == c
        assert dates[0] == c + pd.Timedelta(days=1) and len(dates) == HORIZON
    assert seen[-1][1][-1] == wide.index.max()  # cutoff terakhir pas menghabiskan data ≤ kursor


def test_seasonal_naive_sempurna_di_pola_mingguan():
    wide = panel(weekly_panel())
    assert backtest(seasonal_naive, wide, cutoffs(wide.index.max(), n=3)) == [0.0, 0.0, 0.0]


def test_panel_mengisi_tanggal_hilang_dengan_nol():
    df = weekly_panel(10)
    wide = panel(df[df["date"] != pd.Timestamp("2017-01-05")])
    assert len(wide) == 10 and (wide.loc["2017-01-05"] == 0).all()


def test_close_holidays_nol_di_25_des_dan_1_jan():
    dates = pd.date_range("2016-12-24", "2017-01-02")
    pred = close_holidays(pd.DataFrame({"x": 5.0}, index=dates))
    assert pred["x"].eq(0).tolist() == [d in ("2016-12-25", "2017-01-01") for d in dates.strftime("%Y-%m-%d")]


def test_rmsle_memotong_prediksi_negatif():
    assert rmsle([0, 0], [-5, 0]) == 0.0
    assert np.isclose(rmsle([np.e - 1], [0]), 1.0)
