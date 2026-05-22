# EMNIST Letters — HOG + SVM Classification

Klasifikasi karakter tulisan tangan dari dataset **EMNIST Letters** menggunakan **HOG Feature Extraction** dan **SVM Classifier**.

---

## Dataset

**EMNIST Letters** — subset dari EMNIST dataset yang berisi gambar karakter alfabet tulisan tangan (A–Z).

| Info | Detail |
|------|--------|
| Format | CSV (grayscale pixel values) |
| Ukuran gambar | 28 × 28 piksel (784 fitur) |
| Jumlah kelas | 26 (huruf A–Z) |
| File yang digunakan | `emnist-letters-train.csv` |

### Cara Download Dataset

1. Buka [https://www.kaggle.com/datasets/crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)
2. Login ke akun Kaggle, lalu klik **Download**
3. Ekstrak file, ambil `emnist-letters-train.csv`
4. Letakkan di direktori yang sama dengan notebook

Atau via Kaggle CLI:
```bash
kaggle datasets download -d crawford/emnist
unzip emnist.zip
```

---

## Struktur Project

```
Asesmen Tengah Semester/
├── EMNIST_HOG_SVM_Classification.ipynb   # Notebook utama
├── emnist-letters-train.csv              # Dataset (unduh sendiri)
├── distribusi_kelas.png                  # Hasil EDA: distribusi kelas
├── contoh_sampel_per_kelas.png           # Contoh sampel tiap huruf
├── visualisasi_hog.png                   # Visualisasi fitur HOG
└── confusion_matrix.png                  # Confusion matrix hasil evaluasi
```

---

## Pipeline

```
Load CSV → Balancing (100 sampel/kelas) → EDA → Split 80:20
    → HOG Feature Extraction → Grid Search SVM → Evaluasi
```

1. **Load & Balancing** — Dataset di-*balance* menjadi 100 sampel per kelas = **2.600 data**
2. **EDA** — Visualisasi distribusi kelas dan sampel per huruf
3. **HOG Feature Extraction** — Ekstraksi fitur tepi dan gradien dari tiap gambar
4. **Grid Search SVM** — Pencarian hyperparameter optimal dengan 5-Fold CV
5. **Evaluasi** — Accuracy, Precision, Recall, F1-Score + Confusion Matrix + 10-Fold CV

---

## Parameter Terbaik

**HOG:**
| Parameter | Nilai |
|-----------|-------|
| Orientations | 9 |
| Pixels per Cell | (7, 7) |
| Cells per Block | (2, 2) |
| Jumlah Fitur | 324 per gambar |

**SVM:** hasil Grid Search dari kombinasi:
- `C`: [0.1, 1, 10]
- `gamma`: ['scale', 0.01, 0.001]
- `kernel`: ['linear', 'rbf'] → **RBF terbaik**

---

## Hasil Evaluasi

| Metrik | Training | Testing | CV 10-Fold |
|--------|----------|---------|------------|
| Accuracy | (lihat notebook) | (lihat notebook) | (lihat notebook) |
| Precision | ✓ | ✓ | — |
| Recall | ✓ | ✓ | — |
| F1-Score | ✓ | ✓ | — |

> Terdapat gap antara Training dan Testing Accuracy → indikasi **overfitting ringan**, wajar pada SVM dengan C besar. CV 10-Fold mengkonfirmasi performa konsisten dan tidak bias berlebihan.

**5 kelas dengan akurasi terendah** umumnya huruf yang bentuknya mirip secara visual (contoh: I & L, C & G).

---

## Visualisasi Hasil

| File | Deskripsi |
|------|-----------|
| `distribusi_kelas.png` | Bar chart distribusi 26 kelas setelah balancing |
| `contoh_sampel_per_kelas.png` | Grid contoh satu sampel tiap huruf A–Z |
| `visualisasi_hog.png` | Visualisasi HOG pada 6 huruf: A, E, I, N, T, Z |
| `confusion_matrix.png` | Heatmap confusion matrix 26 × 26 |

---

## Cara Menjalankan

```bash
# Install dependensi
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image

# Jalankan notebook
jupyter notebook EMNIST_HOG_SVM_Classification.ipynb
```

Pastikan `emnist-letters-train.csv` berada di direktori yang sama dengan notebook sebelum menjalankan.

---

## Kesimpulan

- Kombinasi **HOG + SVM (RBF kernel)** efektif untuk klasifikasi 26 kelas karakter tulisan tangan
- HOG menghasilkan **324 fitur** per gambar yang cukup representatif
- Huruf dengan bentuk visual mirip (G, E, I, L) menjadi sumber kesalahan terbesar
- Grid Search berhasil meningkatkan performa dibanding parameter default