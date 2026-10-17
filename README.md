# SalesCast

[![ci](https://github.com/SulthanAndyno/salescast/actions/workflows/ci.yml/badge.svg)](https://github.com/SulthanAndyno/salescast/actions/workflows/ci.yml)
[![retrain](https://github.com/SulthanAndyno/salescast/actions/workflows/retrain.yml/badge.svg)](https://github.com/SulthanAndyno/salescast/actions/workflows/retrain.yml)

Forecast penjualan harian **16 hari ke depan** untuk **1.782 series** (54 toko × 33 kategori) supermarket Favorita,
Ekuador, dengan pipeline CI/CD/CT penuh: retrain otomatis, gate champion vs challenger, registry model, rollback,
dashboard, dan alert. Semua jalan gratis di GitHub Actions + GitHub Pages.

**Dashboard live:** https://sulthanandyno.github.io/salescast/

## Hasil

Metrik: RMSLE (sama dengan kompetisi Kaggle-nya), makin kecil makin baik.

**Backtest walk-forward** (6 cutoff mingguan, 11 Nov – 16 Des 2016, tiap cutoff memprediksi 16 hari):

| Model | RMSLE rata-rata |
|---|---|
| Seasonal naive (hari yang sama minggu lalu) | 0,680 |
| Rata-rata 28 hari terakhir | 0,610 |
| **LightGBM** | **0,528** (menang di 6/6 cutoff) |

**Error realisasi selama simulasi CT** (forecast minggu lalu dinilai terhadap aktual yang baru terungkap; angka terbaru
ada di dashboard):

| Minggu s.d. | LightGBM | Rata-rata 28 hari |
|---|---|---|
| 8 Jan 2017 | 0,583 | 0,589 |
| 15 Jan 2017 | 0,410 | 0,606 |
| 22 Jan 2017 | 0,398 | 0,582 |
| 29 Jan 2017 | 0,389 | 0,524 |

**Per kategori** (holdout 14–29 Jan 2017): paling mudah kategori bervolume besar dan stabil — HOME CARE 0,17,
PRODUCE 0,17, DAIRY 0,18, GROCERY I 0,20. Paling sulit kategori kecil yang sering 0 — LINGERIE 0,63, GROCERY II 0,58,
CELEBRATION 0,57. Selisih terbesar dari baseline ada di FROZEN FOODS (0,36 vs 1,63) dan LIQUOR,WINE,BEER (0,50 vs 1,11):
rata-rata 28 hari ikut menyeret lonjakan Desember ke Januari, LightGBM tidak.

## Cara kerja

```
Kaggle ──download──▶ data/*.parquet (cache Actions)
                         │
   retrain.yml (harian) ─┤ 1. kursor maju 7 hari ("minggu baru datang")
                         │ 2. nilai forecast minggu lalu → reports/metrics.jsonl
                         │ 3. challenger dilatih ≤ kursor−16, diuji di 16 hari terakhir
                         │    vs champion (GitHub Release) & baseline → gate
                         │ 4. lolos → Release model-<kursor>; champion buat forecast 16 hari
                         │ 5. error melonjak → GitHub Issue
                         ▼
   pages.yml ─────────▶ site/data.json + index.html → GitHub Pages
```

- **Kursor waktu.** Data asli berhenti di 15 Agu 2017. Simulasi mulai di 1 Jan 2017; data setelah kursor dianggap belum
  ada. Tiap run memajukan kursor 7 hari, jadi ±31 run sampai data habis.
- **Strategi direct.** Semua fitur penjualan memakai data ≥ 16 hari sebelum tanggal target (lag 16/21/28/35/42/364,
  rata-rata bergerak), sehingga satu model memprediksi H+1..H+16 tanpa prediksi rekursif. Fitur lain: hari gajian,
  libur nasional & lokal (dicocokkan ke kota/provinsi toko), promo, harga minyak (digeser 16 hari).
- **Satu model global** LightGBM untuk semua series, target `log1p(sales)` → L2 loss = optimasi RMSLE.
- **Gate adil.** Challenger dan champion sama-sama dilatih sebelum jendela uji, jadi dua-duanya belum pernah melihatnya.
  Lolos jika RMSLE ≤ champion **dan** ≥ 5% lebih baik dari baseline. Model tidak di-refit setelah lolos; fitur tetap
  dihitung dari data terbaru saat prediksi, jadi yang hilang hanya 16 hari terakhir sebagai contoh training.
- **Test anti-leakage** (`tests/test_features.py`): semua data setelah t diganti 9999, fitur sampai t+16 harus identik.

## Temuan

1. **Data hilang yang menyamar jadi nol.** Simulasi CT minggu pertama membuat LightGBM kalah telak (RMSLE 1,05 vs
   baseline 0,59). Penyebabnya: di sekitar Feb 2014 – Mei 2015, penjualan ±10 kategori (PRODUCE, HOME CARE,
   CELEBRATION, LADIESWEAR, PET SUPPLIES, dll.) tercatat ~0 di ratusan hari (PRODUCE: 300 hari) — data tidak tercatat,
   bukan penjualan anjlok. Jendela training 2 tahun memuat Jan–Mei 2015, jadi model belajar "Januari = anjlok".
   Perbaikan (`features.gaps`): tanggal × kategori yang total nasionalnya < 1% dari rekor sebelumnya dianggap data
   hilang, dikeluarkan dari fitur dan target. Setelahnya Januari 2017 jadi 0,51 vs 0,61, dan dua kategori itu kini
   termudah. Backtest Nov–Des tidak menangkap bug ini; monitoring CT yang menangkapnya.
2. **Hari tutup.** 25 Des (semua toko) dan 1 Jan (hampir semua) tidak ada atau ~0 di data. Prediksi di dua tanggal itu
   dipaksa 0; tanpa itu RMSLE cutoff Desember melonjak ke 1,3–1,7. Seasonal naive tetap rusak sesudahnya karena menyalin
   angka 0 itu ke minggu berikutnya.
3. **Gempa 16 Apr 2016** memicu lonjakan pembelian untuk donasi selama beberapa minggu; 16 Apr – 15 Mei 2016 tidak
   dipakai sebagai target training.
4. **31% nilai penjualan = 0.** Series yang 8 minggu tidak terjual langsung diprediksi 0.

## Batasan (jujur)

- Ini **simulasi**: "data baru" sebenarnya data historis 2017 yang dibuka bertahap. Retrain harian = satu minggu
  simulasi per hari supaya pipeline teruji ±31 kali dalam sebulan.
- `onpromotion` untuk tanggal target diambil dari data (jadwal promo dianggap diketahui sebelumnya, seperti di
  `test.csv` Kaggle). Harga minyak hanya dipakai versi 16 hari lalu.
- Backtest hanya 6 cutoff di musim akhir tahun; angka realisasi CT di dashboard lebih representatif.
- Aturan "total < 1% dari rekor = data hilang" juga menandai musim sepi yang **nyata** (SCHOOL AND OFFICE SUPPLIES
  di luar musim sekolah, BOOKS sejak Mei 2017), sehingga hari-hari itu ikut keluar dari training. Belum diukur
  dampaknya; perbaikannya membedakan gap mendadak (data hilang) dari penurunan bertahap (musiman).
- Belum ada interval prediksi (hanya titik) dan belum ada rekonsiliasi hierarki (toko → total).

## Jalankan lokal

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
| F3 CT mingguan | `salescast/ct.py`, `.github/workflows/retrain.yml` | `python -m salescast.ct` |
| F4 gate + registry + rollback | `salescast/ct.py`, `retrain.yml` | champion di GitHub Releases; rollback: Actions → retrain → Run workflow → isi `rollback_to` |
| F5 dashboard | `salescast/export.py`, `site/index.html`, `pages.yml` | `python -m salescast.export && python -m http.server -d site` |
| F6 alert | `salescast/ct.py` (`alert`), `retrain.yml` | error minggu ini > 1,5× median 8 minggu → GitHub Issue otomatis |

Secret yang dibutuhkan: `KAGGLE_API_TOKEN`. Registry, Pages, dan Issue memakai `GITHUB_TOKEN` bawaan.
