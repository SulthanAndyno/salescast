# SalesCast

Forecast penjualan harian 16 hari ke depan per toko × kategori (Favorita, Kaggle) dengan CI/CD/CT. Rencana: [PLAN.md](PLAN.md).

## Jalankan

```bash
export UV_CACHE_DIR=/workspace/.cache/uv KAGGLEHUB_CACHE=/workspace/.cache/kagglehub
export KAGGLE_API_TOKEN=...   # atau KAGGLE_USERNAME + KAGGLE_KEY
uv sync
uv run python -m salescast.data download   # rules kompetisi harus sudah di-accept di web Kaggle
uv run pytest -q
```

| Fase | File | Perintah |
|---|---|---|
| F0 CI + data + kursor | `.github/workflows/ci.yml`, `salescast/data.py` | `python -m salescast.data download\|cursor\|advance` |
| F1 baseline + backtest | `salescast/backtest.py` | `python -m salescast.backtest` |
| F2 fitur + LightGBM | `salescast/features.py`, `salescast/model.py` | `python -m salescast.model` (±4 menit) |
| F3 CT mingguan | `salescast/ct.py`, `.github/workflows/retrain.yml` | `python -m salescast.ct` (kursor +7 hari → nilai forecast lama → retrain) |
| F4 gate + registry + rollback | `salescast/ct.py`, `retrain.yml` | champion di GitHub Releases; rollback: Actions → retrain → Run workflow → isi `rollback_to` |
| F5 dashboard | `salescast/export.py`, `site/index.html`, `pages.yml` | `python -m salescast.export && python -m http.server -d site` |
