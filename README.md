# Advertising Sales Prediction Web App

Aplikasi web untuk prediksi sales berdasarkan anggaran iklan menggunakan Machine Learning dengan Flask dan Bootstrap.

## Fitur

- **Form Input**: Input anggaran iklan untuk TV, Radio, dan Newspaper
- **Prediksi Real-time**: Menampilkan prediksi sales berdasarkan input
- **Visualisasi**: Kontribusi masing-masing media terhadap sales
- **Model Info**: Informasi detail tentang model yang digunakan
- **Responsive Design**: Menggunakan Bootstrap untuk tampilan yang responsif

## Model Machine Learning

- **Algoritma**: Linear Regression
- **Akurasi**: R² = 89.7%
- **Data Training**: 200 data points
- **Features**: TV, Radio, Newspaper advertising budget
- **Target**: Sales prediction

## Persamaan Model

```
Sales = 4.6812 + 0.0549*TV + 0.1096*Radio + (-0.0062)*Newspaper
```

## Struktur File

```
├── app.py                 # Aplikasi Flask utama
├── model.py              # Training model dan penyimpanan
├── advertising.csv       # Dataset training
├── advertising_model.pkl # Model yang sudah dilatih
├── requirements.txt      # Dependencies Python
├── templates/            # Template HTML
│   ├── base.html        # Template dasar
│   ├── index.html       # Halaman utama dengan form
│   └── model_info.html  # Informasi model
└── README.md            # Dokumentasi
```

## Instalasi dan Penggunaan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training Model (Opsional)

```bash
python model.py
```

### 3. Jalankan Aplikasi Web

```bash
python app.py
```

### 4. Akses Aplikasi

Buka browser dan kunjungi: `http://localhost:5000`

## API Endpoints

### POST /predict
Form submission untuk prediksi sales

**Input:**
- `tv_budget`: Anggaran iklan TV (float)
- `radio_budget`: Anggaran iklan Radio (float)
- `newspaper_budget`: Anggaran iklan Newspaper (float)

### POST /api/predict
API endpoint untuk prediksi (JSON)

**Request:**
```json
{
    "tv_budget": 230.1,
    "radio_budget": 37.8,
    "newspaper_budget": 69.2
}
```

**Response:**
```json
{
    "predicted_sales": 22.1,
    "input": {
        "tv_budget": 230.1,
        "radio_budget": 37.8,
        "newspaper_budget": 69.2
    },
    "model_info": {
        "intercept": 4.6812,
        "tv_coefficient": 0.0549,
        "radio_coefficient": 0.1096,
        "newspaper_coefficient": -0.0062
    }
}
```

### GET /model_info
Halaman informasi detail tentang model

## Interpretasi Hasil

- **TV Coefficient (0.0549)**: Setiap $1000 tambahan iklan TV meningkatkan sales rata-rata 0.055 unit
- **Radio Coefficient (0.1096)**: Setiap $1000 tambahan iklan Radio meningkatkan sales rata-rata 0.110 unit (paling efektif)
- **Newspaper Coefficient (-0.0062)**: Iklan newspaper memiliki dampak negatif atau tidak signifikan

## Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, Bootstrap 5
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Icons**: Font Awesome

## Screenshots

### Halaman Utama
- Form input untuk anggaran iklan
- Validasi input real-time
- Tampilan responsif

### Hasil Prediksi
- Prediksi sales yang akurat
- Breakdown kontribusi setiap media
- Visualisasi yang informatif

### Informasi Model
- Detail koefisien model
- Performa model (MAE, RMSE, R²)
- Interpretasi hasil

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan fitur baru.

## Lisensi

MIT License

