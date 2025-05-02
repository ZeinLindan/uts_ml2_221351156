import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="ann-car-sales-price-prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Car Sales Price Prediction")
st.write("Masukkan informasi kendaraan untuk mendapatkan estimasi harga jual mobil.")

# Form input pengguna
age = st.number_input("Umur", min_value=0.0, max_value=100.0, value=50.0, format="%.6f")
annual_salary = st.number_input("Gaji Tahunan", min_value=0.0, max_value=9999999999.0, value=50.0, format="%.5f")
cc_debt = st.number_input("Hutang Kartu Kredit", min_value=0.0, max_value=9999999999.0, value=50.0, format="%.6f")
net_worth = st.number_input("Kekayaan Bersih", min_value=0.0, max_value=9999999999.0, value=25.0, format="%.4f")

if st.button("Prediksi Harga Mobil"):
    # Preprocessing input
    input_data = np.array([[age, annual_salary, cc_debt, net_worth]])
    input_scaled = scaler_X.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Reverse scaling to get the actual price
    predicted_price = scaler_y.inverse_transform(prediction)[0][0]

    st.success(f"Estimasi harga jual mobil: **Rp {predicted_price:,.5f}**")