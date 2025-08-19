# Spectroscopic-Analyzer

Dokumentasi ringkas untuk memahami struktur proyek, komponen utama, dan cara menjalankan aplikasi.

## Cara Menjalankan

```zsh
# (Opsional) buat dan aktifkan virtualenv
# python -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt
python app/main.py
```

## Struktur Proyek

- `Dockerfile` — build image untuk deployment
- `fly.toml` — konfigurasi Fly.io
- `requirements.txt` — dependensi Python
- `app/` — kode sumber aplikasi
  - `main.py` — entrypoint GUI (membuka Qt `MainWindow`)
  - `model.py` — definisi model dan loader aset (PTH + JSON)
  - `processing.py` — utilitas preprocessing file ASC
  - `core/`
    - `analysis.py` — pipeline analisis end-to-end (parsing, baseline, smoothing, normalisasi, deteksi puncak, pemetaan elemen, validasi, Abel)
  - `ui/`
    - `main_window.py` — logika utama GUI (I/O file, parameter, plotting, ekspor)
    - `worker.py` — worker analisis di thread terpisah (slot `run_preview`/`run_full`, emit `previewFinished`/`fullAnalysisFinished`)
    - `control_panel.py` — panel kontrol kiri terpadu (unggah file, parameter, tombol Pratinjau/Prediksi/Validasi, overlay/ekspor)
    - `results_panel.py` — panel hasil kanan (plot utama, plot zoom tersinkron via region, plot radial, tabel hasil)
    - `panels/` — komponen UI legacy (mungkin tidak lagi dipakai)
      - `left_panel.py` — versi lama panel kiri
      - `results_panel.py` — versi lama panel kanan
  - `worker.py` — implementasi pipeline model alternatif/legacy (tidak dipakai UI saat ini)
- `assets/` — aset model dan peta elemen
  - `informer_multilabel_model.pth` — bobot model
  - `element-map-18a.json` — peta elemen terhadap grid panjang gelombang
  - `wavelengths_grid.json` — grid panjang gelombang target
- `example-asc/` — contoh file ASC untuk uji cepat

## Komponen & Fungsi Utama

- `app/core/analysis.py`
  - `run_full_analysis(input_data) -> dict` — menerima input dari UI (termasuk `asc_content`, parameter, dan `analysis_mode`), menghasilkan hasil analisis:
    - `preprocess`: sinyal terproses + puncak tanpa anotasi/tabel
    - `predict/validate`: anotasi puncak, tabel prediksi/validasi, metrik, opsi profil radial (Abel)

- `app/model.py`
  - `InformerModel` dan layer pendukung (PyTorch)
  - `als_baseline_correction(y, lam, p, niter)` — koreksi baseline ALS
  - `load_assets()` — memuat model dan file JSON aset

- `app/processing.py`
  - `prepare_asc_data(asc_content_string) -> np.ndarray` — parsing ASC ke array

- `app/ui/main_window.py`
  - `MainWindow(QMainWindow)` — mengelola alur UI + worker:
    - Aksi: `start_preprocess`, `start_prediction`, `start_validation`
    - Input: `get_input_data`
    - Hasil: `update_results`, `export_results_to_xlsx`
    - Interaksi: zoom/crosshair/overlay, debounce slider

- `app/ui/worker.py`
  - `Worker(QObject)` — menyediakan dua jalur eksekusi:
    - `run_preview` → emit `previewFinished` (pra-pemrosesan cepat untuk pratinjau)
    - `run_full` → emit `fullAnalysisFinished` (prediksi/validasi penuh)

- `app/ui/control_panel.py`
  - `ControlPanel(QWidget)` — panel kontrol kiri terpadu dengan sinyal `previewRequested` dan `analysisRequested`.

- `app/ui/results_panel.py`
  - `ResultsPanel(QWidget)` — plot utama + zoom (LinearRegionItem), plot radial, dan tabel hasil.

## Alur Kerja Singkat

1. Pengguna memilih file ASC dan mengatur parameter di panel kiri.
2. `MainWindow` mengirim:
  - `previewRequested` ke `Worker.run_preview` untuk pratinjau cepat (pra-pemrosesan saja)
  - `analyzeRequested` ke `Worker.run_full` untuk prediksi/validasi penuh
3. `Worker` menjalankan analisis sesuai mode dan mengembalikan hasil melalui sinyal `previewFinished` / `fullAnalysisFinished`.
4. `ResultsPanel.update_preview` / `update_full_results` menggambar plot/tabel sesuai mode dan opsi yang dipilih.

## Catatan

- Jika aset tidak tersedia, pemetaan elemen mungkin terbatas; preprocessing tetap berjalan.
- Abel inversion membutuhkan `PyAbel` (opsional).



Tentu. Ini adalah ide yang sangat baik untuk menstrukturkan semua ide kita ke dalam sebuah roadmap yang jelas.
Berikut adalah draf Rencana Pengembangan (Roadmap) dalam format Markdown. Ini bisa Anda simpan sebagai file ROADMAP.md di dalam proyek Anda dan bisa menjadi panduan untuk sprint pengembangan selanjutnya, baik untuk Anda maupun untuk memandu Copilot.
Roadmap Pengembangan Aplikasi Spektroskopi
Dokumen ini menguraikan target pengembangan untuk mengubah aplikasi dari prototipe analisis menjadi perangkat lunak ilmiah yang matang dan produktif. Pengembangan dibagi menjadi beberapa sprint, masing-masing dengan tujuan yang jelas.
Sprint 1: Peningkatan Produktivitas & Pengalaman Pengguna (UX)
Tujuan Sprint: Meningkatkan efisiensi kerja pengguna secara drastis dan memberikan fondasi untuk fitur yang lebih kompleks dengan memungkinkan manajemen data dan parameter yang lebih baik.
Fitur 1: Batch Processing (Pemrosesan Massal)
Sebagai seorang peneliti, saya ingin dapat memilih satu folder berisi banyak file .asc dan memproses semuanya secara otomatis, sehingga saya tidak perlu menganalisis file satu per satu.
Tugas Implementasi (Perintah untuk Copilot):
Buat QDialog baru yang memungkinkan pengguna memilih folder input dan folder output.
Buat fungsi yang memindai semua file .asc di folder input.
Untuk setiap file, jalankan fungsi run_full_analysis di background thread.
Kumpulkan hasil utama (misalnya, daftar elemen yang terdeteksi per file) ke dalam satu DataFrame pandas.
Setelah semua selesai, tampilkan tabel ringkasan di UI dan simpan DataFrame tersebut sebagai satu file Excel ringkasan di folder output.
Fitur 2: Preset Parameter
Sebagai seorang analis, saya ingin bisa menyimpan dan memuat satu set lengkap pengaturan parameter, sehingga saya bisa dengan cepat beralih antara konfigurasi untuk jenis sampel yang berbeda (misalnya, "Analisis Baja" vs. "Analisis Tanah").
Tugas Implementasi (Perintah untuk Copilot):
Tambahkan dua QPushButton di panel kontrol: "Simpan Preset" dan "Muat Preset".
Fungsi "Simpan Preset" akan memanggil get_input_data untuk mengumpulkan semua nilai parameter, lalu menggunakan QFileDialog untuk menyimpannya sebagai file JSON.
Fungsi "Muat Preset" akan membuka file JSON dan mengisi kembali semua nilai QLineEdit, QCheckBox, dll., di UI.
Sprint 2: Kedalaman Analisis & Validasi Ilmiah
Tujuan Sprint: Memperkaya analisis dengan data referensi eksternal dan kemampuan dekonvolusi puncak, meningkatkan kepercayaan pada hasil.
Fitur 1: Integrasi Database Spektral (NIST)
Sebagai seorang ilmuwan, saya ingin bisa menampilkan garis-garis emisi referensi dari database NIST di atas spektrum saya, sehingga saya bisa secara visual memvalidasi puncak yang diidentifikasi oleh model AI.
Tugas Implementasi (Perintah untuk Copilot):
Buat parser sederhana untuk membaca file data LIBS dari NIST (yang sudah di-format sebagai CSV). Data ini akan berisi (nama_elemen, panjang_gelombang).
Di panel kontrol, tambahkan QComboBox yang diisi dengan semua elemen unik dari database.
Saat pengguna memilih elemen dari ComboBox, gambar pg.InfiniteLine vertikal di plot untuk setiap panjang gelombang referensi elemen tersebut.
Buat mekanisme untuk menghapus atau menyembunyikan garis referensi ini.
Fitur 2: Peak Fitting & Deconvolution
Sebagai seorang spektroskopis, saya ingin bisa memilih area dengan puncak yang tumpang tindih dan memecahnya menjadi beberapa puncak individual (Gaussian/Lorentzian), sehingga saya bisa menganalisis setiap kontribusi secara terpisah.
Tugas Implementasi (Perintah untuk Copilot):
Gunakan LinearRegionItem yang sudah ada. Saat sebuah region dipilih, tambahkan tombol "Fit Puncak di Region".
Saat tombol ditekan, ambil data di dalam region tersebut.
Gunakan scipy.optimize.curve_fit untuk mencocokkan (fit) satu atau beberapa fungsi profil puncak (misalnya, jumlah dari beberapa fungsi Gaussian) ke data.
Tampilkan kurva hasil fitting dan puncak-puncak individual yang terdekonvolusi di plot zoom.
Sprint 3: Analisis Kuantitatif & Pelaporan Profesional
Tujuan Sprint: Mengembangkan kemampuan aplikasi dari kualitatif ("apa") menjadi kuantitatif ("berapa banyak") dan menghasilkan laporan yang siap dipublikasikan.
Fitur 1: Analisis Kuantitatif (Kemometrik)
Sebagai seorang analis industri, saya ingin membangun model kalibrasi dari sampel standar untuk memprediksi konsentrasi elemen dalam sampel yang tidak diketahui, sehingga saya bisa melakukan kontrol kualitas.
Tugas Implementasi (Perintah untuk Copilot):
Ini adalah fitur besar yang memerlukan mode baru di aplikasi.
Buat UI baru untuk "Manajemen Kalibrasi" di mana pengguna bisa memuat beberapa file .asc dan memasukkan konsentrasi yang diketahui untuk setiap sampel.
Gunakan scikit-learn untuk membangun model regresi (misalnya PLSRegression) dari data spektral dan konsentrasi.
Setelah model dibuat, pengguna bisa memuat sampel baru, dan aplikasi akan menggunakan model tersebut untuk memprediksi dan menampilkan konsentrasinya.
Fitur 2: Generator Laporan (PDF/HTML)
Sebagai seorang manajer lab, saya ingin menghasilkan laporan satu halaman yang ringkas dalam format PDF untuk setiap analisis, sehingga saya bisa dengan mudah mengarsipkan dan membagikan hasilnya.
Tugas Implementasi (Perintah untuk Copilot):
Tambahkan tombol "Buat Laporan PDF".
Saat ditekan, kumpulkan semua hasil saat ini: gambar dari plot utama, tabel hasil, dan metrik kuantitatif.
Gunakan library seperti ReportLab atau fpdf2 untuk menyusun informasi ini secara terstruktur di dalam sebuah file PDF.
Buka dialog "Simpan File" untuk menyimpan PDF yang dihasilkan.