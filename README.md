# Laporan Proyek Machine Learning: Prediksi Risiko Penyakit Stroke Menggunakan Komparasi Algoritma Decision Tree, KNN, dan Random Forest dengan Pendekatan SMOTE

## 1. Identitas Proyek
* **Judul Proyek:** Prediksi Risiko Penyakit Stroke Menggunakan Komparasi Algoritma Decision Tree, K-Nearest Neighbors (KNN), dan Random Forest dengan Pendekatan SMOTE
* **Nama / NIM:** [Dzikri Khopi Slamet Susandi (2406169), Wildan Syaeful Millah Albatani (2306118)]
* **Mata Kuliah:** Kecerdasan Buatan (UAS)

---

## 2. Domain Proyek (Latar Belakang)
Stroke merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Penyakit ini terjadi ketika suplai darah ke bagian otak terganggu atau berkurang, sehingga jaringan otak tidak mendapatkan oksigen dan nutrisi yang cukup. Deteksi dini terhadap risiko stroke sangat penting dilakukan dalam upaya pencegahan dan penanganan medis yang tepat agar dapat menyelamatkan nyawa pasien (Karyadiputra et al., 2025). 

Seiring berkembangnya teknologi di bidang kesehatan, penerapan Kecerdasan Buatan (AI) khususnya *Machine Learning* telah menjadi solusi mutakhir untuk mengklasifikasi rekam medis pasien guna mendeteksi risiko penyakit secara dini. Berbagai penelitian terdahulu menunjukkan bahwa algoritma berbasis pohon seperti *Random Forest* sangat andal dalam memprediksi penyakit stroke (Azhar et al., 2022; Fadli & Saputra, 2023). Namun, masalah yang sering muncul pada data rekam medis adalah ketidakseimbangan kelas (*imbalanced data*), di mana jumlah pasien sehat jauh lebih banyak daripada pasien stroke. Hal ini diatasi dengan menggunakan teknik penyampelan data *Synthetic Minority Over-sampling Technique* (SMOTE) untuk meningkatkan performa deteksi pada kelas minoritas (Aryabima et al., 2025; Nabila & Pamungkas, 2025).

## 3. Business Understanding
**Problem Statements:**
1. Keterlambatan deteksi risiko stroke karena evaluasi medis konvensional yang memakan waktu.
2. Tingginya angka kesalahan prediksi pasien berisiko (*false negative*) pada model deteksi akibat dataset kesehatan yang didominasi oleh pasien normal (data tidak seimbang).

**Goals (Tujuan):**
1. Membangun model *Machine Learning* yang mampu memprediksi risiko stroke berdasarkan faktor gaya hidup dan metrik klinis pasien.
2. Membandingkan kinerja tiga algoritma: *Decision Tree, K-Nearest Neighbors (KNN), dan Random Forest* untuk menemukan model yang paling optimal dalam meminimalisir pasien stroke yang terlewat (memaksimalkan nilai Recall).
3. Menerapkan teknik SMOTE pada proses pelatihan untuk menangani ketidakseimbangan kelas.

## 4. Data Understanding
Dataset yang digunakan berasal dari Kaggle, yaitu **Healthcare Dataset Stroke Data**. Dataset ini mencakup 5.110 observasi pasien dengan 11 kolom atribut prediktor dan 1 variabel target:
* **Fitur Kategorikal:** `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
* **Fitur Numerikal:** `age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`.
* **Target Label:** `stroke` (0 = Tidak Stroke, 1 = Mengalami Stroke).

Berdasarkan *Exploratory Data Analysis (EDA)*, ditemukan bahwa data kelas target sangat tidak seimbang (*imbalanced*). Kelompok pasien tidak stroke mencapai sekitar 95,1%, sedangkan penderita stroke hanya sekitar 4,9%.

## 5. Data Preparation
Tahapan pemrosesan data (Data Preprocessing) yang dilakukan adalah sebagai berikut:
1. **Data Cleaning:** Menghapus kolom `id` karena tidak relevan secara prediktif. Menghapus 1 baris kategori gender `Other` karena dianggap sebagai *outlier* kategorik. Serta mengisi 201 nilai yang kosong (*missing values*) pada kolom `bmi` menggunakan nilai median (nilai tengah).
2. **Encoding Kategorik:** Menerapkan *Label Encoding* untuk fitur biner/ordinal (`gender`, `ever_married`, `Residence_type`) dan *One-Hot Encoding* untuk fitur nominal yang memiliki lebih dari dua kategori (`work_type`, `smoking_status`).
3. **Normalisasi (Scaling):** Menerapkan *StandardScaler* pada fitur numerik (`age`, `avg_glucose_level`, `bmi`) agar rentang nilainya seragam dan tidak mendominasi perhitungan jarak pada algoritma KNN.
4. **Data Splitting & SMOTE:** Dataset dibagi menjadi 80% data latih (7.776 sampel setelah SMOTE) dan 20% data uji (1.022 sampel asli). Teknik **SMOTE** diaplikasikan **hanya** pada data latih agar distribusi kelas menjadi seimbang (3.888 kelas 0 vs 3.888 kelas 1) guna mencegah *Data Leakage*.

## 6. Modeling
Proyek ini mengeksplorasi tiga algoritma klasifikasi:
1. **Decision Tree:** Algoritma pohon keputusan tunggal yang dilatih dengan kedalaman maksimal (`max_depth = 5`) untuk mencegah *overfitting*.
2. **K-Nearest Neighbors (KNN):** Algoritma berbasis jarak spasial (*Euclidean distance*) dengan penentuan kelas berdasarkan 5 tetangga terdekat (`n_neighbors = 5`).
3. **Random Forest:** Algoritma *ensemble* tingkat lanjut yang membangun banyak pohon keputusan (200 *trees*, `max_depth = 8`) untuk menekan varians dan meningkatkan stabilitas prediksi.

## 7. Evaluation (Hasil Evaluasi)
Pengujian performa dilakukan pada data uji (1.022 data pasien tanpa manipulasi SMOTE). Karena dataset asli bersifat medis dan imbalanced, fokus utama evaluasi ditekankan pada metrik **Recall** (kemampuan AI menemukan pasien stroke) dan **F1-Score** (keseimbangan tebakan kelas minoritas).

Berikut adalah ringkasan hasil evaluasi ketiga model:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | 74.66% | 12.27% | **68.00%** | **20.80%** |
| **Decision Tree** | 73.87% | 11.93% | **68.00%** | 20.30% |
| **K-Nearest Neighbors** | 78.38% | 10.23% | 44.00% | 16.60% |

**Kesimpulan:**
Meskipun K-Nearest Neighbors (KNN) memiliki akurasi global tertinggi (78.38%), model tersebut gagal mendeteksi mayoritas pasien stroke sesungguhnya (Recall hanya 44%). Sebaliknya, model **Random Forest** muncul sebagai model **terbaik** dengan nilai F1-Score tertinggi (20.80%) dan mampu mengenali **68.00%** dari total pasien berisiko stroke yang sebenarnya (Recall yang sangat krusial dalam dunia medis). Secara keseluruhan, tujuan proyek tercapai dan teknik SMOTE berhasil memaksa algoritma berbasis pohon untuk lebih sensitif mendeteksi kasus stroke.

---

## 8. Daftar Pustaka
Aryabima, M. I., Rusdah, Roeswidiah, R., & Pudoli, A. (2025). Deteksi dini penyakit stroke pada data tidak seimbang menggunakan SMOTE dan Random Forest. *Jurnal TICOM: Technology of Information and Communication, 13*(3), 141-146.

Azhar, Y., Firdausy, A. K., & Amelia, P. J. (2022). Perbandingan algoritma klasifikasi data mining untuk prediksi penyakit stroke. *SINTECH JOURNAL, 5*(2), 191-197.

Fadli, M., & Saputra, R. A. (2023). Klasifikasi dan evaluasi performa model Random Forest untuk prediksi stroke. *JT: Jurnal Teknik, 12*(2), 72-80.

Karyadiputra, E., Setiawan, A., Ramadhani, B., & Purnomo, I. I. (2025). Aplikasi prediksi resiko penyakit stroke. *Technologia: Jurnal Ilmiah, 16*(2), 141-149.

Nabila, H. A., & Pamungkas, E. W. (2025). Perbandingan algoritma machine learning: SVM, Random Forest, dan XGBoost untuk prediksi stroke. *RABIT: Jurnal Teknologi dan Sistem Informasi Univrab, 10*(2), 1098-1110.
