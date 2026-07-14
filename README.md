# Aplikasi Prediksi Sentimen Saham IDX80 dengan LSTM

Aplikasi Streamlit ini memprediksi sentimen artikel berita saham IDX80 menggunakan model LSTM. Alur aplikasi disesuaikan dengan laporan skripsi: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` dibuat khusus untuk deployment aplikasi agar instalasi di Streamlit Cloud lebih ringan.

## Training Model

Training dilakukan terpisah dari aplikasi Streamlit. Jalankan script ini hanya ketika dataset atau model perlu diperbarui:

```bash
pip install -r requirements-train.txt
python train_model.py
```

Secara default script akan memakai dataset `idx80_3000.csv` dan menyimpan hasil training ke folder `artifacts/`.

Contoh jika ingin mengatur epoch atau dataset:

```bash
python train_model.py --dataset idx80_3000.csv --epochs 20 --batch-size 32
```

File yang dihasilkan:

- `artifacts/lstm_sentiment_model.keras`
- `artifacts/tokenizer.pkl`
- `artifacts/label_encoder.pkl`
- `artifacts/config.json`
- `artifacts/metrics.json`

## Aplikasi Streamlit

`app.py` hanya digunakan untuk prediksi. File ini memuat model, tokenizer, label encoder, dan konfigurasi dari folder `artifacts/`. Di dalam `app.py` tidak ada proses training ulang, pembuatan arsitektur LSTM, `model.fit()`, atau `model.save()`.

## Format Dataset Training

Gunakan file CSV dengan minimal kolom:

- `judul`: judul artikel berita
- `isi_berita`: isi artikel berita
- `sentiment`: label sentimen, misalnya `Naik`, `Turun`, atau `Netral`

Dataset dari notebook sebelumnya memakai encoding `latin1`, jadi aplikasi menyediakan pilihan encoding saat upload. Saat training, label akan diseragamkan menjadi `Positif`, `Negatif`, dan `Netral` agar sesuai dengan laporan.

## Alur Aplikasi

1. Buka tab `Prediksi`.
2. Gunakan mode `Artikel Tunggal` untuk memprediksi satu berita.
3. Gunakan mode `Batch CSV` untuk memprediksi banyak artikel sekaligus.
4. Jika file `idx80_3000.csv` tersedia di folder aplikasi, mode batch dapat langsung memakai dataset tersebut tanpa upload manual.
5. Buka tab `Informasi Model` untuk melihat konfigurasi model yang sedang digunakan.

Model dan artefak pendukung tersimpan di folder `artifacts/`.

Catatan: training model tidak ditampilkan sebagai halaman utama aplikasi agar pengguna tidak perlu melatih ulang model saat melakukan prediksi. Jika dataset diperbarui dan model perlu dilatih ulang, jalankan proses training melalui `train_model.py` atau notebook `LSTM_Skripsi.ipynb`, lalu simpan kembali artefak model ke folder `artifacts/`.
