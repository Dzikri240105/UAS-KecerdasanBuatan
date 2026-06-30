# UAS Kecerdasan Buatan - Prediksi Penyakit Stroke

Repositori ini dibuat untuk memenuhi tugas UAS mata kuliah Kecerdasan Buatan. Laporan lengkap proyek ini (mulai dari Latar Belakang hingga Referensi) dapat diakses pada file **[Laporan_uas.md](Laporan_uas.md)**.

## Langkah Pengerjaan Proyek

Berikut adalah tahapan pengerjaan model *Machine Learning* yang dilakukan dalam proyek ini:

1. **Pengumpulan Data (*Data Collection*):** Mengunduh *Healthcare Dataset Stroke Data* dari Kaggle dan menyimpannya di dalam folder `data/`.
2. **Pemahaman Data (*Data Understanding*):** Melakukan analisis awal (EDA) untuk memeriksa tipe data, jumlah baris, dan distribusi kelas target. Ditemukan bahwa data mengalami ketidakseimbangan kelas (*imbalanced data*) yang sangat tinggi (95.12% data pasien tidak stroke dan 4.88% data pasien stroke).
3. **Persiapan Data (*Data Preparation*):** * Menghapus kolom `id` karena tidak relevan untuk pemodelan.
   * Melakukan *encoding* (mengubah data teks menjadi numerik) pada fitur kategorik.
   * Melakukan standardisasi skala pada fitur numerik (`age`, `bmi`, `avg_glucose_level`).
   * Membagi data menjadi data latih (*training*) sebesar 80% dan data uji (*testing*) sebesar 20%.
4. **Penanganan Data Tidak Seimbang (*Handling Imbalanced Data*):** Menerapkan teknik **SMOTE** (*Synthetic Minority Over-sampling Technique*) khusus pada data latih untuk menyeimbangkan jumlah sampel pasien stroke.
5. **Pemodelan (*Modeling*):** Membangun, melatih, dan menguji 3 algoritma klasifikasi secara komparatif:
   * **Decision Tree**
   * **K-Nearest Neighbors (KNN)**
   * **Random Forest**
6. **Evaluasi Model (*Evaluation*):** Menguji performa ketiga model menggunakan *Confusion Matrix* dengan berfokus pada metrik *Recall* demi meminimalkan kesalahan diagnosis medis (*False Negative*).

---
*Seluruh kode program dapat dilihat pada file: `uas_model.ipynb`*
