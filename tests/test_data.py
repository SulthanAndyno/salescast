import pandas as pd

from salescast.data import END, advance, load, read_cursor


def test_advance_maju_7_hari_dan_berhenti_di_akhir_data(tmp_path):
    cur = tmp_path / "cursor.txt"
    cur.write_text("2017-01-01\n")
    assert advance(path=cur) == pd.Timestamp("2017-01-08")
    assert read_cursor(cur) == pd.Timestamp("2017-01-08")

    cur.write_text("2017-08-12\n")
    assert advance(path=cur) == END
    assert advance(path=cur) == END


def test_load_membuang_data_setelah_kursor(tmp_path):
    dates = pd.date_range("2017-01-01", periods=10)
    pd.DataFrame({"date": dates, "sales": range(10)}).to_parquet(tmp_path / "train.parquet")
    df = load("train", cursor=pd.Timestamp("2017-01-05"), data=tmp_path)
    assert df["date"].max() == pd.Timestamp("2017-01-05")
    assert len(df) == 5
