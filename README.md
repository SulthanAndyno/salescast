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
