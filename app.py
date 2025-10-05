from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

app = Flask(__name__)

# Memuat model yang sudah dilatih
try:
    with open('advertising_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model berhasil dimuat!")
except FileNotFoundError:
    print("File model tidak ditemukan. Pastikan model.py sudah dijalankan terlebih dahulu.")
    model = None

@app.route('/')
def index():
    """Halaman utama dengan form input"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk melakukan prediksi"""
    try:
        # Mendapatkan data dari form
        tv_budget = float(request.form['tv_budget'])
        radio_budget = float(request.form['radio_budget'])
        newspaper_budget = float(request.form['newspaper_budget'])
        
        # Validasi input
        if tv_budget < 0 or radio_budget < 0 or newspaper_budget < 0:
            return render_template('index.html', 
                                 error="Anggaran tidak boleh negatif!",
                                 tv_budget=tv_budget,
                                 radio_budget=radio_budget,
                                 newspaper_budget=newspaper_budget)
        
        # Membuat array input untuk prediksi
        input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
        
        # Melakukan prediksi
        if model is not None:
            prediction = model.predict(input_data)[0]
            
            # Menghitung kontribusi masing-masing media
            tv_contribution = model.coef_[0] * tv_budget
            radio_contribution = model.coef_[1] * radio_budget
            newspaper_contribution = model.coef_[2] * newspaper_budget
            base_sales = model.intercept_
            
            # Menyiapkan data untuk ditampilkan
            result = {
                'predicted_sales': round(prediction, 2),
                'tv_budget': tv_budget,
                'radio_budget': radio_budget,
                'newspaper_budget': newspaper_budget,
                'tv_contribution': round(tv_contribution, 2),
                'radio_contribution': round(radio_contribution, 2),
                'newspaper_contribution': round(newspaper_contribution, 2),
                'base_sales': round(base_sales, 2),
                'total_contribution': round(tv_contribution + radio_contribution + newspaper_contribution, 2)
            }
            
            return render_template('index.html', result=result)
        else:
            return render_template('index.html', 
                                 error="Model tidak tersedia. Pastikan model.py sudah dijalankan.")
            
    except ValueError:
        return render_template('index.html', 
                             error="Masukkan angka yang valid untuk semua field!")
    except Exception as e:
        return render_template('index.html', 
                             error=f"Terjadi kesalahan: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi (JSON)"""
    try:
        data = request.get_json()
        
        tv_budget = float(data['tv_budget'])
        radio_budget = float(data['radio_budget'])
        newspaper_budget = float(data['newspaper_budget'])
        
        input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
        
        if model is not None:
            prediction = model.predict(input_data)[0]
            
            return jsonify({
                'predicted_sales': round(prediction, 2),
                'input': {
                    'tv_budget': tv_budget,
                    'radio_budget': radio_budget,
                    'newspaper_budget': newspaper_budget
                },
                'model_info': {
                    'intercept': round(model.intercept_, 4),
                    'tv_coefficient': round(model.coef_[0], 4),
                    'radio_coefficient': round(model.coef_[1], 4),
                    'newspaper_coefficient': round(model.coef_[2], 4)
                }
            })
        else:
            return jsonify({'error': 'Model tidak tersedia'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/pairplot')
def generate_pairplot():
    """Generate pairplot image"""
    try:
        df = pd.read_csv('advertising.csv')
        
        # Create pairplot with seaborn style
        plt.style.use('default')
        fig = sns.pairplot(df, height=2.5, aspect=1, 
                           plot_kws={'alpha': 0.7, 's': 20},
                           diag_kws={'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 0.5})
        
        # Customize the plot
        fig.fig.suptitle('Pairplot: Hubungan Antar Variabel', y=1.02, fontsize=16, fontweight='bold')
        
        # Save to bytes
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        print(f"Error generating pairplot: {e}")
        return "Error generating pairplot", 500

@app.route('/model_info')
def model_info():
    """Halaman informasi model"""
    if model is not None:
        # Memuat data untuk scatter plot dan pairplot
        try:
            df = pd.read_csv('advertising.csv')
            X = df[['TV', 'Radio', 'Newspaper']]
            y = df['Sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
            predictions = model.predict(X_test)
            
            # Konversi ke list untuk JSON serialization
            y_test_list = y_test.tolist()
            predictions_list = predictions.tolist()
            
            # Data untuk pairplot interaktif
            pairplot_data = {
                'TV': df['TV'].tolist(),
                'Radio': df['Radio'].tolist(),
                'Newspaper': df['Newspaper'].tolist(),
                'Sales': df['Sales'].tolist()
            }
            
        except Exception as e:
            print(f"Error loading data: {e}")
            y_test_list = []
            predictions_list = []
            pairplot_data = {
                'TV': [], 'Radio': [], 'Newspaper': [], 'Sales': []
            }
        
        model_data = {
            'intercept': round(model.intercept_, 4),
            'coefficients': {
                'TV': round(model.coef_[0], 4),
                'Radio': round(model.coef_[1], 4),
                'Newspaper': round(model.coef_[2], 4)
            },
            'equation': f"Sales = {model.intercept_:.4f} + {model.coef_[0]:.4f}*TV + {model.coef_[1]:.4f}*Radio + {model.coef_[2]:.4f}*Newspaper",
            'y_test': y_test_list,
            'predictions': predictions_list,
            'pairplot_data': pairplot_data
        }
        return render_template('model_info.html', model_data=model_data)
    else:
        return render_template('model_info.html', error="Model tidak tersedia")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
