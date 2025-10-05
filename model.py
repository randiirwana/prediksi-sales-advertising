import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Membaca data dari file CSV
df = pd.read_csv('advertising.csv')
print("Data Advertising:")
print(df.head())
print(f"\nShape data: {df.shape}")
print(f"\nInfo data:")
print(df.info())

# Visualisasi hubungan antar variabel
sns.pairplot(df)
plt.show()

# Memisahkan fitur (X) dan target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
# Membagi data menjadi data latih (training) dan data uji (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Membuat instance dari model
model = LinearRegression()
# Melatih model menggunakan data training
model.fit(X_train, y_train)
# Melakukan prediksi pada data test
predictions = model.predict(X_test)
# Visualisasi hasil prediksi vs nilai aktual
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Sales Aktual (y_test)")
plt.ylabel("Sales Prediksi (predictions)")
plt.title("Sales Aktual vs. Sales Prediksi")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2) # Garis ideal
plt.show()
# Evaluasi performa model
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
# Menampilkan intersep dan koefisien
print(f"\nIntercept: {model.intercept_}")
print("\nKoefisien untuk setiap fitur:")
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coef_df)

# Menampilkan persamaan regresi
print(f"\nPersamaan Regresi:")
print(f"Sales = {model.intercept_:.4f} + {model.coef_[0]:.4f}*TV + {model.coef_[1]:.4f}*Radio + {model.coef_[2]:.4f}*Newspaper")

# Menyimpan model untuk digunakan di aplikasi web
import pickle
with open('advertising_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nModel telah disimpan ke 'advertising_model.pkl'")