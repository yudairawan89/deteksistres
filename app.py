import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Konfigurasi Halaman Streamlit
# ================================
st.set_page_config(page_title="Deteksi Tingkat Stres", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #e74c3c;'>🧠 Sistem Deteksi Tingkat Stres Mahasiswa</h1>
    <p style='text-align: center;'>Deteksi Realtime dari Data IoT</p>
    <hr>
""", unsafe_allow_html=True)

# ================================
# Load model dan scaler
# ================================
model = joblib.load("model_stres_terbaik.joblib")
scaler = joblib.load("scaler_stres.joblib")

# Mapping label numerik ke stres
label_mapping = {
    0: "Anxious",
    1: "Calm",
    2: "Relaxed",
    3: "Tense"
}
color_mapping = {
    "Anxious": "#e74c3c",
    "Calm": "#2ecc71",
    "Relaxed": "#3498db",
    "Tense": "#f39c12"
}

# ================================
# Fungsi Ambil Data dari Google Sheet
# ================================
def load_latest_data_from_sheets():
    sheet_csv_url = "https://docs.google.com/spreadsheets/d/1Sc961SwCUZ3TExI04YhSRELJSL8nQsQ4VAfsLtV8WSQ/export?format=csv"
    df = pd.read_csv(sheet_csv_url)
    return df.iloc[-1]

# ================================
# Fungsi Prediksi
# ================================
def prediksi_stres(input_data):
    data_scaled = scaler.transform([input_data])
    pred = model.predict(data_scaled)[0]
    return label_mapping[pred]

# ================================
# Deteksi Realtime dari Google Sheet
# ================================
st.subheader("🔄 Deteksi Stres Realtime dari IoT")
if st.button("Deteksi Stres"):
    try:
        latest = load_latest_data_from_sheets()
        input_data = [
            float(latest["Suhu (°C)"]),
            float(latest["SpO2 (%)"]),
            float(latest["HeartRate (BPM)"])
        ]
        hasil = prediksi_stres(input_data)

        # ===== Tampilan Data Fisiologis Realtime =====
        st.markdown(f"""
            <div style="background-color:#f4f4f4; padding:20px; border-radius:10px; margin-bottom:20px;">
                <h4 style='color:#333;'>📋 Data Fisiologis</h4>
                <p><b>Suhu Tubuh:</b> {input_data[0]} °C</p>
                <p><b>Oksigen dalam Darah (SpO2):</b> {input_data[1]} %</p>
                <p><b>Detak Jantung:</b> {input_data[2]} BPM</p>
            </div>
        """, unsafe_allow_html=True)

        # ===== Tampilan Hasil Deteksi =====
        st.markdown(f"""
            <div style='background-color:{color_mapping[hasil]}; padding:20px; border-radius:10px; text-align:center;'>
                <h2 style='color:white;'>Tingkat Stres: {hasil}</h2>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Gagal membaca data realtime: {e}")

# ================================
# Pengujian Manual
# ================================
st.subheader("✍️ Uji Data Manual")
with st.form("manual_form"):
    suhu = st.number_input("Suhu Tubuh (°C)", min_value=30.0, max_value=45.0, step=0.1)
    spo2 = st.number_input("SpO2 (%)", min_value=50.0, max_value=100.0, step=0.1)
    hr = st.number_input("Heart Rate (BPM)", min_value=30.0, max_value=200.0, step=1.0)
    submitted = st.form_submit_button("Deteksi Manual")
    if submitted:
        input_data = [suhu, spo2, hr]
        hasil = prediksi_stres(input_data)

        # ===== Tampilan Manual Input =====
        st.markdown(f"""
            <div style="background-color:#f4f4f4; padding:20px; border-radius:10px; margin-bottom:20px;">
                <h4 style='color:#333;'>📋 Data yang Diuji</h4>
                <p><b>Suhu Tubuh:</b> {input_data[0]} °C</p>
                <p><b>Oksigen dalam Darah (SpO2):</b> {input_data[1]} %</p>
                <p><b>Detak Jantung:</b> {input_data[2]} BPM</p>
            </div>
        """, unsafe_allow_html=True)

        # ===== Tampilan Hasil Deteksi Manual =====
        st.markdown(f"""
            <div style='background-color:{color_mapping[hasil]}; padding:20px; border-radius:10px; text-align:center;'>
                <h2 style='color:white;'>Tingkat Stres: {hasil}</h2>
            </div>
        """, unsafe_allow_html=True)
