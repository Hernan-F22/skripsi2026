# Aplikasi Prediksi Sentimen Saham IDX80 dengan LSTM

Aplikasi Streamlit ini memprediksi sentimen artikel berita saham IDX80 menggunakan model LSTM. Alur aplikasi disesuaikan dengan laporan skripsi: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format Dataset Training

Gunakan file CSV dengan minimal kolom:

- `judul`: judul artikel berita
- `isi_berita`: isi artikel berita
- `sentiment`: label sentimen, misalnya `Naik`, `Turun`, atau `Netral`

Dataset dari notebook sebelumnya memakai encoding `latin1`, jadi aplikasi menyediakan pilihan encoding saat upload. Saat training, label akan diseragamkan menjadi `Positif`, `Negatif`, dan `Netral` agar sesuai dengan laporan.

## Alur Aplikasi

1. Buka tab `Training`.
2. Upload dataset CSV.
3. Pilih kolom judul, isi berita, dan label sentimen.
4. Atur hyperparameter LSTM.
5. Klik `Latih dan Simpan Model`.
6. Buka tab `Prediksi` untuk memprediksi artikel baru.
7. Gunakan tab `Batch CSV` untuk memprediksi banyak artikel sekaligus.

Model dan artefak pendukung akan tersimpan di folder `artifacts/`.
