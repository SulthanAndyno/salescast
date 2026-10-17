"""Satu langkah continuous training: "minggu baru datang".

1. Kursor maju 7 hari → 7 hari aktual baru terungkap.
2. Forecast minggu lalu dinilai terhadap aktual itu → baris baru di reports/metrics.jsonl (data monitoring).
3. Challenger dilatih di data ≤ kursor − 16, diuji di 16 hari terakhir bersama champion & baseline (gate).
   Champion juga dilatih ≤ cutoff yang lebih lama, jadi dua-duanya belum pernah melihat 16 hari itu → adil.
4. Model yang menang gate (champion) membuat forecast 16 hari ke depan → reports/forecast.parquet.

Registry: file model ada di GitHub Releases (diunduh/diunggah oleh retrain.yml), penunjuk champion di
reports/champion.json yang di-commit.

python -m salescast.ct                     # satu langkah
python -m salescast.ct rollback VERSION    # pakai versi lama (models/champion.txt sudah diunduh workflow)
python -m salescast.ct alert               # cetak pesan alert kalau error melonjak (kosong = normal)
"""

import json
import re
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from salescast.backtest import HORIZON, close_holidays, moving_average, panel, rmsle
from salescast.data import DATA, END, advance, load, read_cursor
from salescast.features import SERIES
from salescast.model import context, fit, predict

REPORTS = DATA.parent / "reports"
CHAMPION = REPORTS / "champion.json"
CHAMPION_MODEL = DATA.parent / "models" / "champion.txt"
MODELS = ("lightgbm", "moving_average_28")
MARGIN = 0.95  # challenger wajib ≥ 5% lebih baik dari baseline
ALERT_FACTOR = 1.5  # error minggu ini > 1,5× median 8 minggu sebelumnya → buka GitHub Issue
DASHBOARD = "https://sulthanandyno.github.io/salescast/"


def score(forecast: pd.DataFrame, wide: pd.DataFrame, prev: pd.Timestamp, new: pd.Timestamp) -> dict:
    """RMSLE forecast lama di hari-hari yang baru terungkap (prev, new]."""
    f = forecast[(forecast["date"] > prev) & (forecast["date"] <= new)]
    actual = wide.stack(SERIES).rename("actual").rename_axis(["date", *SERIES])
    f = f.join(actual, on=["date", *SERIES], how="inner")
    row = {"cursor": str(new.date()), "days": int(f["date"].nunique())}
    row |= {m: round(rmsle(f["actual"], f[m]), 4) for m in MODELS}
    return row


def gate(holdout: dict) -> list[str]:
    """Alasan challenger DITOLAK (kosong = lolos)."""
    chal, base = holdout["challenger"], holdout["moving_average_28"]
    why = []
    if chal > MARGIN * base:
        why.append(f"challenger {chal} tidak ≥5% lebih baik dari moving_average_28 {base}")
    if "champion" in holdout and chal > holdout["champion"]:
        why.append(f"challenger {chal} > champion {holdout['champion']}")
    return why


def load_champion() -> lgb.Booster | None:
    return lgb.Booster(model_file=str(CHAMPION_MODEL)) if CHAMPION_MODEL.exists() else None


def train_and_gate(ctx: dict, wide: pd.DataFrame, cursor: pd.Timestamp) -> tuple[lgb.Booster | None, dict]:
    cutoff = cursor - pd.Timedelta(days=HORIZON)
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON)

    def s(pred: pd.DataFrame) -> float:
        return round(rmsle(wide.loc[dates], close_holidays(pred)), 4)

    chal = fit(ctx, wide, cutoff)
    holdout = {
        "challenger": s(predict(chal, ctx, wide, dates)),
        "moving_average_28": s(moving_average(wide.loc[:cutoff], dates)),
    }
    champ = load_champion()
    if champ is not None:
        try:
            holdout["champion"] = s(predict(champ, ctx, wide, dates))
        # fitur berubah sejak champion dilatih → champion tidak bisa dinilai, gate hanya vs baseline
        except (KeyError, ValueError, lgb.basic.LightGBMError) as e:
            print("champion tidak kompatibel dengan fitur sekarang:", e)
            champ = None

    why = gate(holdout)
    if not why:
        version = f"model-{cursor.date()}"
        CHAMPION_MODEL.parent.mkdir(exist_ok=True)
        chal.save_model(str(CHAMPION_MODEL))
        CHAMPION.write_text(
            json.dumps({"version": version, "train_cutoff": str(cutoff.date()), "holdout": holdout}) + "\n"
        )
        champ = chal
    current = json.loads(CHAMPION.read_text())["version"] if CHAMPION.exists() else None
    return champ, {"holdout": holdout, "promoted": not why, "rejected_because": why, "champion": current}


def forecast(model: lgb.Booster | None, ctx: dict, wide: pd.DataFrame, cursor: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(cursor + pd.Timedelta(days=1), periods=HORIZON)
    ma = moving_average(wide, dates)
    # belum ada model yang lolos gate → yang dipakai baseline (aman), bukan model yang ditolak
    preds = {"lightgbm": predict(model, ctx, wide, dates) if model is not None else ma, "moving_average_28": ma}
    long = {m: close_holidays(p).stack(SERIES) for m, p in preds.items()}
    return pd.DataFrame(long).rename_axis(["date", *SERIES]).reset_index()


def write_forecast(model: lgb.Booster | None, ctx: dict, wide: pd.DataFrame, cursor: pd.Timestamp) -> None:
    out = forecast(model, ctx, wide, cursor)
    out.to_parquet(REPORTS / "forecast.parquet", index=False)
    print(f"forecast {out['date'].min().date()} … {out['date'].max().date()}, total {np.round(out['lightgbm'].sum())}")


def step() -> None:
    REPORTS.mkdir(exist_ok=True)
    fpath = REPORTS / "forecast.parquet"
    cursor = read_cursor()
    row = {"cursor": str(cursor.date())}
    if fpath.exists():
        if cursor >= END:
            print(f"kursor sudah di akhir data ({END.date()}), simulasi selesai")
            return
        new = advance()
        row = score(pd.read_parquet(fpath), panel(load("train", new)), cursor, new)
        cursor = new

    wide = panel(load("train", cursor))
    ctx = context(cursor)
    model, gated = train_and_gate(ctx, wide, cursor)
    row |= gated
    with open(REPORTS / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    write_forecast(model, ctx, wide, cursor)


def alert(rows: list[dict]) -> str | None:
    """Pesan alert kalau error realisasi minggu terakhir melonjak; None kalau normal."""
    if not rows or "lightgbm" not in rows[-1]:
        return None  # baris terakhir bukan penilaian mingguan (mis. rollback)
    hist = [r["lightgbm"] for r in rows if "lightgbm" in r]
    last, prev = hist[-1], hist[-9:-1]
    if len(prev) < 3 or last <= ALERT_FACTOR * float(np.median(prev)):
        return None
    r = rows[-1]
    return (
        f"RMSLE LightGBM minggu s.d. {r['cursor']}: **{last}** "
        f"(> {ALERT_FACTOR}× median {len(prev)} minggu sebelumnya, {np.median(prev):.4f}). "
        f"Baseline minggu ini: {r['moving_average_28']}.\n\n"
        f"- Dashboard: {DASHBOARD}\n"
        "- Cek data minggu ini (anomali / data hilang) sebelum menyalahkan model.\n"
        "- Kalau champion yang bermasalah: Actions → retrain → Run workflow → isi `rollback_to` dengan versi lama "
        "(daftar di halaman Releases)."
    )


def rollback(version: str) -> None:
    if not re.fullmatch(r"model-\d{4}-\d{2}-\d{2}", version) or not CHAMPION_MODEL.exists():
        sys.exit(f"versi {version} tidak valid atau models/champion.txt belum diunduh")
    CHAMPION.write_text(json.dumps({"version": version, "rollback": True}) + "\n")
    cursor = read_cursor()
    wide = panel(load("train", cursor))
    write_forecast(load_champion(), context(cursor), wide, cursor)
    with open(REPORTS / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"cursor": str(cursor.date()), "rollback": version}) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["rollback"] and len(sys.argv) == 3:
        rollback(sys.argv[2])
    elif sys.argv[1:] == ["alert"]:
        rows = [json.loads(line) for line in (REPORTS / "metrics.jsonl").read_text().splitlines() if line]
        print(alert(rows) or "", end="")
    else:
        step()
