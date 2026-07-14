# Panduan Publish ke Streamlit Community Cloud

## File yang Wajib Di-upload ke GitHub

Pastikan file dan folder berikut masuk ke repository:

- `app.py`
- `train_model.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`
- `idx80_3000.csv`
- `artifacts/config.json`
- `artifacts/lstm_weights.npz`
- `artifacts/tokenizer.json`

Folder `artifacts/` wajib ikut di-upload karena aplikasi membutuhkan bobot LSTM NumPy, tokenizer JSON, dan konfigurasi model untuk halaman prediksi.
File `idx80_3000.csv` wajib ikut di-upload jika ingin fitur Batch CSV langsung memakai dataset bawaan tanpa upload manual.

Catatan: Streamlit Cloud menjalankan `app.py` sebagai aplikasi prediksi. Script `train_model.py` hanya digunakan untuk training offline ketika model perlu diperbarui.

`requirements.txt` sengaja dibuat ringan untuk Streamlit Cloud. Aplikasi tidak memakai TensorFlow/Keras saat deployment. Dependency training seperti `tensorflow-cpu`, `scikit-learn`, dan `matplotlib` dipindahkan ke `requirements-train.txt`.

## File yang Tidak Wajib Di-upload

File berikut tidak perlu di-upload karena hanya data lokal atau lampiran:

- `hasil_prediksi_sentimen_idx80.csv`
- `lampiran_hasil_prediksi_sentimen_idx80_50_baris.csv`
- `Lampiran_Hasil_Prediksi_Sentimen_IDX80_50_Baris.docx`
- `__pycache__/`

## Langkah Publish

1. Buat repository baru di GitHub.
2. Upload semua file proyek dari folder `D:\Skripsi 2026\Python`.
3. Pastikan folder `artifacts/` ikut masuk ke GitHub.
4. Buka https://share.streamlit.io.
5. Pilih `New app`.
6. Pilih repository GitHub.
7. Isi `Main file path` dengan:

```text
app.py
```

8. Klik `Deploy`.

## Jika Error

Jika error muncul, cek bagian `Manage app` lalu buka `Logs`. Error yang paling sering:

- `No such file or directory: artifacts/...`: folder `artifacts/` belum ikut di-upload.
- `ModuleNotFoundError`: package belum ada di `requirements.txt`.
- error TensorFlow/Keras: pastikan `runtime.txt` berisi `python-3.11`.
