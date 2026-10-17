# SalesCast

[![ci](https://github.com/SulthanAndyno/salescast/actions/workflows/ci.yml/badge.svg)](https://github.com/SulthanAndyno/salescast/actions/workflows/ci.yml)
[![retrain](https://github.com/SulthanAndyno/salescast/actions/workflows/retrain.yml/badge.svg)](https://github.com/SulthanAndyno/salescast/actions/workflows/retrain.yml)
[![pages](https://github.com/SulthanAndyno/salescast/actions/workflows/pages.yml/badge.svg)](https://sulthanandyno.github.io/salescast/)

Forecasting penjualan harian 16 hari ke depan untuk 1.782 deret waktu (54 toko × 33 kategori produk) jaringan
supermarket Corporación Favorita, Ekuador. Pipeline mencakup retraining terjadwal, evaluasi champion/challenger,
model registry, rollback, monitoring akurasi, dan alert yang seluruhnya berjalan di GitHub Actions.

Dashboard: https://sulthanandyno.github.io/salescast/

![Dashboard](docs/dashboard.png)

## Hasil

| Model | RMSLE backtest |
|---|---|
| Seasonal naive (minggu lalu) | 0,680 |
| Moving average 28 hari | 0,610 |
| LightGBM | 0,528 |

Backtest walk-forward pada 6 cutoff mingguan (11 Nov s.d. 16 Des 2016) dengan horizon 16 hari. LightGBM unggul di
seluruh cutoff.

Error realisasi selama simulasi, yaitu forecast yang dievaluasi setelah data aktualnya tersedia:

| Minggu s.d. | LightGBM | Moving average 28 hari |
|---|---|---|
| 8 Jan 2017 | 0,583 | 0,589 |
| 15 Jan 2017 | 0,410 | 0,606 |
| 22 Jan 2017 | 0,398 | 0,582 |
| 29 Jan 2017 | 0,389 | 0,524 |

Data terbaru tersedia di dashboard. Per kategori, error terendah ada pada kategori bervolume tinggi (HOME CARE dan
PRODUCE 0,17, DAIRY 0,18) dan tertinggi pada kategori bervolume rendah dengan banyak nilai nol (LINGERIE 0,63,
GROCERY II 0,58).

## Arsitektur

| Workflow | Pemicu | Tahapan |
|---|---|---|
| `ci.yml` | push, pull request | lint (ruff), test (pytest) |
| `retrain.yml` | harian 08.00 WIB, manual | evaluasi forecast sebelumnya, training challenger, gate, rilis model, forecast baru, alert, commit hasil |
| `pages.yml` | setelah retrain, perubahan `site/` | ekspor data dan deploy dashboard ke GitHub Pages |

Karena dataset bersifat historis, kedatangan data baru disimulasikan dengan kursor waktu di `data/cursor.txt`. Data
setelah tanggal kursor diperlakukan sebagai belum tersedia. Kursor dimulai 1 Januari 2017 dan maju 7 hari setiap run.

Model yang lolos gate dirilis sebagai GitHub Release dengan tag `model-<tanggal>`, dan `reports/champion.json` menunjuk
versi yang sedang aktif. Rollback dilakukan melalui input `rollback_to` pada workflow retrain.

RMSLE realisasi setiap minggu dicatat di `reports/metrics.jsonl`. Jika nilainya melebihi 1,5 kali median 8 minggu
sebelumnya, workflow membuka GitHub Issue secara otomatis.

## Metodologi

### Model

Satu model LightGBM global untuk seluruh deret dengan target `log1p(sales)`, sehingga loss L2 ekuivalen dengan
optimasi RMSLE.

### Fitur

Menggunakan strategi direct multi-horizon: seluruh fitur penjualan hanya memakai data minimal 16 hari sebelum tanggal
target, sehingga satu model dapat memprediksi H+1 hingga H+16 tanpa prediksi rekursif.

- Lag 16, 21, 28, 35, 42, dan 364 hari, rata-rata bergerak 7/14/28/56 hari, standar deviasi 28 hari
- Kalender: hari, tanggal, bulan, dan hari gajian sektor publik (tanggal 15 dan akhir bulan)
- Hari libur nasional, regional, dan lokal yang dipetakan ke kota dan provinsi toko
- Jumlah item promo, harga minyak (lag 16 hari), tipe dan cluster toko

### Validasi

Test anti-leakage memastikan fitur hingga t+16 tidak berubah ketika seluruh data setelah t dimodifikasi.

Pada setiap run, challenger dilatih hingga 16 hari sebelum kursor, kemudian dievaluasi bersama champion dan baseline
pada 16 hari terakhir. Kedua model belum pernah melihat periode tersebut saat training. Challenger dipromosikan jika
RMSLE-nya tidak lebih buruk dari champion dan minimal 5% lebih baik dari baseline.

## Catatan Data

- Pada Februari 2014 s.d. Mei 2015, sekitar 10 kategori (antara lain PRODUCE, HOME CARE, dan CELEBRATION) tercatat
  mendekati nol selama ratusan hari akibat data yang tidak tercatat. Tanpa penanganan, model mempelajari pola
  penurunan palsu di bulan Januari dan RMSLE minggu pertama simulasi mencapai 1,05 (baseline 0,59). Masalah ini
  terdeteksi melalui monitoring mingguan, bukan melalui backtest. Fungsi `features.gaps` kini mengeluarkan tanggal
  ketika total nasional suatu kategori di bawah 1% dari nilai tertinggi sebelumnya, dan RMSLE Januari 2017 turun
  menjadi 0,51.
- Tanggal 25 Desember tidak tercatat dan penjualan 1 Januari mendekati nol. Prediksi pada kedua tanggal tersebut
  diset 0 dan keduanya dikecualikan dari fitur.
- Gempa 16 April 2016 memicu lonjakan pembelian selama beberapa minggu, sehingga periode 16 April s.d. 15 Mei 2016
  dikecualikan dari target training.
- Sebanyak 31% observasi bernilai 0. Deret tanpa penjualan selama 8 minggu diprediksi 0.

## Batasan

- Retraining berjalan atas data historis 2017 yang dibuka bertahap; satu run harian merepresentasikan satu minggu.
- Data promo pada periode forecast diasumsikan sudah diketahui sebelumnya, sesuai format `test.csv` kompetisi.
- Aturan deteksi data hilang juga menandai penurunan musiman yang nyata, misalnya SCHOOL AND OFFICE SUPPLIES dan BOOKS.
- Forecast berupa point estimate tanpa interval prediksi.

## Menjalankan

Prasyarat: Python 3.12 atau lebih baru, [uv](https://docs.astral.sh/uv/), dan API token Kaggle. Aturan kompetisi
harus diterima terlebih dahulu di halaman Kaggle.

```bash
uv sync
export KAGGLE_API_TOKEN=<token>
uv run python -m salescast.data download   # unduh data ke data/*.parquet
uv run pytest -q
uv run python -m salescast.model           # backtest (sekitar 4 menit)
uv run python -m salescast.ct              # satu langkah retraining
uv run python -m salescast.export          # data dashboard ke site/data.json
```

Secret repository yang diperlukan hanya `KAGGLE_API_TOKEN`. Release, Issue, dan Pages menggunakan `GITHUB_TOKEN`.

## Struktur Proyek

```
salescast/
├── data.py        unduh data, kursor waktu
├── backtest.py    metrik, baseline, backtest walk-forward
├── features.py    feature engineering, deteksi data hilang
├── model.py       training dan inferensi LightGBM
├── ct.py          retraining, gate, rollback, alert
└── export.py      ekspor data dashboard
site/              dashboard statis
reports/           metrik, forecast, champion aktif (diperbarui workflow)
tests/
```

## Sumber Data

Data berasal dari kompetisi Kaggle Store Sales - Time Series Forecasting yang memuat data penjualan Corporación
Favorita. Data tidak didistribusikan ulang di repositori ini dan penggunaannya mengikuti
[aturan kompetisi](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules).

> Alexis Cook, DanB, inversion, and Ryan Holbrook. Store Sales - Time Series Forecasting.
> https://kaggle.com/competitions/store-sales-time-series-forecasting, 2021. Kaggle.
