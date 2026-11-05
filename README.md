# Sistem Rekomendasi Wisata Berbasis Konten: Sentence-BERT + Filter Musim

Penulis: Ar'raffi Abqori Nur Azizi, Isma' Yafa Nur Zamzami Ramadhani, Muhammad Maulana Khanif, Andika Kamaluddin.
Isi NIM tiga anggota terakhir di sel judul notebook dan daftar TEAM pada app.py.

## Deploy ke Streamlit Cloud (app selalu hidup, tanpa tunnelling)
1. Buka https://share.streamlit.io → Continue with GitHub (akun `robet31`).
2. Create app → Deploy a public app from GitHub → pilih repo `robet31/proyek-sbert-cbf`.
3. Main file: `app.py`, lalu Deploy.
4. Model `firqaaa/indo-sentence-bert-base` diunduh otomatis saat pertama jalan (~2–3 menit; reload bila belum selesai).

## Langkah (menjalankan lokal)
1. Jalankan `CBF_SBERT_Rekomendasi_Wisata.ipynb` (Run all). GPU disarankan. Data diambil dari berkas lokal atau Google Drive.
2. Sel terakhir membuat `hasil_cbf.zip`. Kirim isinya (terutama `hasil_untuk_artikel.json`, `figures/`, dan CSV) untuk menyusun ulang artikel.
3. Aplikasi: `pip install -r requirements.txt`, salin `artifacts/` hasil notebook ke folder ini, lalu `streamlit run app.py`.

## Pengaturan penting (sel konfigurasi notebook)
- `SBERT_AKTIF = ["indo"]`: sistem utama. Tambahkan `"minilm"` hanya bila ingin model SBERT kedua.
- `MODEL_UTAMA = "indo"`.
- `RUN_PEMBANDING = True`: TF-IDF sebagai acuan agar angka SBERT dapat ditafsirkan (bukan tujuan penelitian).
- Bila model utama gagal dimuat pada data asli, notebook berhenti dengan pesan jelas.

## Status pengujian
Seluruh kode diuji hingga selesai pada data tiruan dengan pengganti `sentence_transformers` tiruan. SBERT asli dan data asli belum dijalankan oleh penyusun kode ini.
