import pandas as pd

from salescast.backtest import panel
from salescast.ct import score


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
