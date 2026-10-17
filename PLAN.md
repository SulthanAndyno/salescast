# SalesCast: rencana fase

Forecast penjualan harian 16 hari ke depan per **toko × kategori produk** (54 toko × 33 kategori = 1.782 series)
pakai data supermarket Favorita, dengan CI/CD/CT yang menyimulasikan "data baru datang tiap minggu".

## Data

- Kaggle: `store-sales-time-series-forecasting` (kompetisi; **wajib klik "I Understand and Accept" rules di web Kaggle
  dulu**, kalau tidak, API-nya 403).
- File: `train.csv` (2013-01-01 → 2017-08-15), `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`.
  `test.csv` tidak dipakai (tidak ada label).
- Metrik: **RMSLE** (sama dengan kompetisinya). Target dilatih dalam bentuk `log1p(sales)`, jadi L2 loss = optimasi RMSLE.

## Ide inti CT: kursor waktu

Data dianggap "diketahui" hanya sampai tanggal di `data/cursor.txt` (mulai `2017-01-01`).
Setiap run CT, kursor maju 7 hari, lalu model di-retrain dengan data ≤ kursor. Dari 2017-01 sampai 2017-08
ada sekitar 32 "minggu baru", cukup buat grafik monitoring yang meyakinkan.

## Keputusan desain (sengaja dibuat sederhana)

| Keputusan | Alasan |
|---|---|
| Satu model LightGBM global untuk semua series | lebih kuat & jauh lebih simpel daripada 1.782 model |
| Strategi *direct*: semua lag ≥ 16 hari | satu model langsung memprediksi H+1..H+16 tanpa prediksi rekursif |
| Baseline: seasonal naive (nilai hari yang sama minggu lalu) | model wajib mengalahkannya; kalau tidak, modelnya tidak berguna |
| Dashboard statis di GitHub Pages | tidak perlu server; forecast diekspor jadi JSON |
| Registry di GitHub Releases | tidak perlu token tambahan; workflow pakai `GITHUB_TOKEN` bawaan |

---

## F0: Setup + CI
- `uv` project, `ruff`, `pytest`; `.github/workflows/ci.yml` → `uv run ruff check && uv run pytest -q`.
- `salescast/data.py`: download lewat Kaggle API (secret `KAGGLE_USERNAME`, `KAGGLE_KEY`), simpan sebagai parquet,
  baca/maju-kan kursor.
- **Selesai kalau:** `python -m salescast.data download` jalan lokal & di Actions (data di-cache `actions/cache`).

## F1: Baseline + backtest walk-forward
- `salescast/backtest.py`: potong data di beberapa cutoff (mis. 6 cutoff mingguan sebelum kursor), prediksi 16 hari,
  hitung RMSLE per cutoff & rata-rata.
- Baseline seasonal naive & moving average 28 hari.
- **Selesai kalau:** tabel RMSLE baseline tercetak; test memastikan split tidak pernah memakai data > cutoff.

## F2: Fitur + LightGBM
- `salescast/features.py`:
  - lag 16/21/28/35/364, rolling mean/std 7/14/28 (digeser ≥16 hari)
  - kalender: hari dalam minggu, tanggal, **hari gajian** (tgl 15 & akhir bulan; sektor publik Ekuador)
  - libur nasional/regional/lokal dicocokkan ke kota/provinsi toko
  - `onpromotion` (sudah diketahui di masa depan → boleh dipakai), harga minyak di-lag
  - flag gempa 2016-04-16 (lonjakan anomali, sebaiknya di-mask/di-flag)
- `salescast/model.py`: train LightGBM, simpan model + metadata (cutoff, RMSLE, versi fitur).
- **Test wajib: anti-leakage.** Ubah nilai `sales` setelah tanggal t, fitur untuk tanggal ≤ t+16 harus tetap sama.
- **Selesai kalau:** RMSLE backtest LightGBM < baseline di semua cutoff.

## F3: CT, simulasi minggu baru
- `retrain.yml`: cron mingguan + `workflow_dispatch`. Langkahnya: maju-kan kursor 7 hari → retrain → evaluasi → commit
  `cursor.txt` & metrik.
- Sebelum retrain, **skor forecast minggu lalu terhadap aktual yang baru "terungkap"**, lalu tulis ke
  `reports/metrics.jsonl` (ini data monitoring).
- **Selesai kalau:** 3 run manual berturut-turut memajukan kursor & menambah 3 baris metrik.

## F4: Gate + registry + rollback
- Challenger dilatih ≤ (kursor − 16), dievaluasi di 16 hari terakhir; champion dievaluasi di jendela yang sama.
- Promosikan jika: RMSLE challenger ≤ champion **dan** ≥ 5% lebih baik dari moving average 28 hari.
  Tidak di-refit: champion lama juga dilatih ≤ cutoff-nya sendiri, jadi dua-duanya belum pernah melihat jendela uji
  (fitur tetap dihitung dari data terbaru saat prediksi). Lolos → GitHub Release `model-<kursor>`.
- Rollback: Actions → retrain → Run workflow → `rollback_to`.
- **Selesai kalau:** test gate menolak challenger yang kalah dari champion/baseline dan menerima yang menang.

## F5: CD dashboard
- `salescast/export.py` → `site/forecast.json` (forecast 16 hari + 60 hari aktual terakhir per series) &
  `site/metrics.json`.
- `site/index.html` statis: pilih toko & kategori → grafik aktual vs prediksi; grafik RMSLE per minggu.
- Deploy ke GitHub Pages otomatis setelah model dipromosikan.
- **Selesai kalau:** URL Pages menampilkan forecast terbaru setelah run CT.

## F6: Monitoring + laporan
- Alert: jika RMSLE realisasi minggu ini > 1,5× median 8 minggu terakhir → Action membuka GitHub Issue.
- README: arsitektur, cara jalan, tabel hasil (baseline vs model), feature importance, dan **catatan jujur**
  (batas simulasi kursor, series yang sering 0, efek gempa, kategori yang paling susah diprediksi).
- **Selesai kalau:** README punya angka nyata dari run, bukan placeholder.

---

## Sengaja tidak dikerjakan (tambah kalau perlu)
- API FastAPI: dashboard statis sudah cukup buat demo; tambah kalau ada konsumen yang butuh query live.
- Model per-horizon / ensemble / deep learning (N-BEATS, TFT): tambah setelah LightGBM mentok.
- MLflow: metadata JSON + HF Hub cukup untuk sekarang; tambah kalau eksperimen mulai banyak.
- Hierarchical reconciliation (toko → total): tambah kalau mau forecast agregat yang konsisten.
