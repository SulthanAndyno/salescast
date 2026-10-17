import pandas as pd

from salescast.backtest import panel
from salescast.ct import alert, gate, score


def test_score_hanya_menilai_hari_yang_baru_terungkap():
    dates = pd.date_range("2017-01-01", periods=20)
    wide = panel(pd.DataFrame({"date": dates, "store_nbr": 1, "family": "A", "sales": 10.0}))
    forecast = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            # tepat di hari 3..9 (yang terungkap), meleset jauh di luar jendela
            "lightgbm": [10.0 if 3 <= i <= 9 else 999.0 for i in range(20)],
            "moving_average_28": 10.0,
        }
    )
    row = score(forecast, wide, prev=dates[2], new=dates[9])
    assert row == {"cursor": "2017-01-10", "days": 7, "lightgbm": 0.0, "moving_average_28": 0.0}


def test_gate():
    # tanpa champion: cukup ≥5% lebih baik dari baseline
    assert gate({"challenger": 0.50, "moving_average_28": 0.60}) == []
    assert len(gate({"challenger": 0.58, "moving_average_28": 0.60})) == 1
    # dengan champion: wajib tidak lebih buruk dari champion juga
    assert gate({"challenger": 0.45, "champion": 0.45, "moving_average_28": 0.60}) == []
    assert len(gate({"challenger": 0.46, "champion": 0.45, "moving_average_28": 0.60})) == 1
    assert len(gate({"challenger": 0.59, "champion": 0.45, "moving_average_28": 0.60})) == 2


def test_alert_hanya_saat_error_melonjak():
    weeks = [{"cursor": f"2017-01-{d:02d}", "lightgbm": 0.40, "moving_average_28": 0.6} for d in range(1, 9)]
    assert alert(weeks) is None
    spike = weeks + [{"cursor": "2017-01-09", "lightgbm": 0.70, "moving_average_28": 0.6}]
    assert "0.7" in alert(spike)
    assert alert(spike + [{"cursor": "2017-01-09", "rollback": "model-2017-01-01"}]) is None
    assert alert(weeks[:2] + spike[-1:]) is None  # histori < 3 minggu: belum bisa menilai
