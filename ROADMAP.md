# Roadmap Pengembangan Aplikasi Spektroskopi

Dokumen ini menguraikan target pengembangan untuk mengubah aplikasi dari prototipe analisis menjadi perangkat lunak ilmiah yang matang dan produktif. Pengembangan dibagi menjadi beberapa sprint, masing-masing dengan tujuan yang jelas.

## Sprint 1: Peningkatan Produktivitas & Pengalaman Pengguna (UX)
Tujuan Sprint: Meningkatkan efisiensi kerja pengguna secara drastis dan memberikan fondasi untuk fitur yang lebih kompleks dengan memungkinkan manajemen data dan parameter yang lebih baik.

### Fitur 1: Batch Processing (Pemrosesan Massal)
Sebagai seorang peneliti, saya ingin dapat memilih satu folder berisi banyak file .asc dan memproses semuanya secara otomatis, sehingga saya tidak perlu menganalisis file satu per satu.

Tugas Implementasi:
- Buat QDialog baru yang memungkinkan pengguna memilih folder input dan folder output.
- Buat fungsi yang memindai semua file .asc di folder input.
- Untuk setiap file, jalankan fungsi `run_full_analysis` di background thread.
- Kumpulkan hasil utama (misalnya, daftar elemen yang terdeteksi per file) ke dalam satu DataFrame pandas.
- Setelah semua selesai, tampilkan tabel ringkasan di UI dan simpan DataFrame tersebut sebagai satu file Excel ringkasan di folder output.

### Fitur 2: Preset Parameter
Sebagai seorang analis, saya ingin bisa menyimpan dan memuat satu set lengkap pengaturan parameter, sehingga saya bisa dengan cepat beralih antara konfigurasi untuk jenis sampel yang berbeda (misalnya, "Analisis Baja" vs. "Analisis Tanah").

Tugas Implementasi:
- Tambahkan dua QPushButton di panel kontrol: "Simpan Preset" dan "Muat Preset".
- Fungsi "Simpan Preset" akan memanggil `get_input_data` untuk mengumpulkan semua nilai parameter, lalu menggunakan `QFileDialog` untuk menyimpannya sebagai file JSON.
- Fungsi "Muat Preset" akan membuka file JSON dan mengisi kembali semua nilai QLineEdit, QCheckBox, dll., di UI.

## Sprint 2: Kedalaman Analisis & Validasi Ilmiah
Tujuan Sprint: Memperkaya analisis dengan data referensi eksternal dan kemampuan dekonvolusi puncak, meningkatkan kepercayaan pada hasil.

### Fitur 1: Integrasi Database Spektral (NIST)
Sebagai seorang ilmuwan, saya ingin bisa menampilkan garis-garis emisi referensi dari database NIST di atas spektrum saya, sehingga saya bisa secara visual memvalidasi puncak yang diidentifikasi oleh model AI.

Tugas Implementasi:
- Buat parser sederhana untuk membaca file data LIBS dari NIST (yang sudah di-format sebagai CSV). Data ini akan berisi `(nama_elemen, panjang_gelombang)`.
- Di panel kontrol, tambahkan QComboBox yang diisi dengan semua elemen unik dari database.
- Saat pengguna memilih elemen dari ComboBox, gambar `pg.InfiniteLine` vertikal di plot untuk setiap panjang gelombang referensi elemen tersebut.
- Buat mekanisme untuk menghapus atau menyembunyikan garis referensi ini.

### Fitur 2: Peak Fitting & Deconvolution
Sebagai seorang spektroskopis, saya ingin bisa memilih area dengan puncak yang tumpang tindih dan memecahnya menjadi beberapa puncak individual (Gaussian/Lorentzian), sehingga saya bisa menganalisis setiap kontribusi secara terpisah.

Tugas Implementasi:
- Gunakan `LinearRegionItem` yang sudah ada. Saat sebuah region dipilih, tambahkan tombol "Fit Puncak di Region".
- Saat tombol ditekan, ambil data di dalam region tersebut.
- Gunakan `scipy.optimize.curve_fit` untuk mencocokkan (fit) satu atau beberapa fungsi profil puncak (misalnya, jumlah dari beberapa fungsi Gaussian) ke data.
- Tampilkan kurva hasil fitting dan puncak-puncak individual yang terdekonvolusi di plot zoom.

## Sprint 3: Analisis Kuantitatif & Pelaporan Profesional
Tujuan Sprint: Mengembangkan kemampuan aplikasi dari kualitatif ("apa") menjadi kuantitatif ("berapa banyak") dan menghasilkan laporan yang siap dipublikasikan.

### Fitur 1: Analisis Kuantitatif (Kemometrik)
Sebagai seorang analis industri, saya ingin membangun model kalibrasi dari sampel standar untuk memprediksi konsentrasi elemen dalam sampel yang tidak diketahui, sehingga saya bisa melakukan kontrol kualitas.

Tugas Implementasi:
- Ini adalah fitur besar yang memerlukan mode baru di aplikasi.
- Buat UI baru untuk "Manajemen Kalibrasi" di mana pengguna bisa memuat beberapa file .asc dan memasukkan konsentrasi yang diketahui untuk setiap sampel.
- Gunakan scikit-learn untuk membangun model regresi (misalnya `PLSRegression`) dari data spektral dan konsentrasi.
- Setelah model dibuat, pengguna bisa memuat sampel baru, dan aplikasi akan menggunakan model tersebut untuk memprediksi dan menampilkan konsentrasinya.

### Fitur 2: Generator Laporan (PDF/HTML)
Sebagai seorang manajer lab, saya ingin menghasilkan laporan satu halaman yang ringkas dalam format PDF untuk setiap analisis, sehingga saya bisa dengan mudah mengarsipkan dan membagikan hasilnya.

Tugas Implementasi:
- Tambahkan tombol "Buat Laporan PDF".
- Saat ditekan, kumpulkan semua hasil saat ini: gambar dari plot utama, tabel hasil, dan metrik kuantitatif.
- Gunakan library seperti ReportLab atau `fpdf2` untuk menyusun informasi ini secara terstruktur di dalam sebuah file PDF.
- Buka dialog "Simpan File" untuk menyimpan PDF yang dihasilkan.
