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
    <p style='text-align: center;'>Berbasis Data Fisiologis Realtime dari Google Sheets</p>
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
# Fungsi Ambil Data dari Google Sheet (CSV)
# ================================
def load_latest_data_from_sheets():
    sheet_csv_url = "https://docs.google.com/spreadsheets/d/1Sc961SwCUZ3TExI04YhSRELJSL8nQsQ4VAfsLtV8WSQ/export?format=csv"
    df = pd.read_csv(sheet_csv_url)
    return df

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
st.subheader("🔄 Deteksi Stres Realtime dari Google Sheets")
if st.button("Deteksi Stres (Realtime)"):
    try:
        df = load_latest_data_from_sheets()
        latest = df.iloc[-1]
        input_data = [
            float(latest["Suhu (°C)"]),
            float(latest["SpO2 (%)"]),
            float(latest["HeartRate (BPM)"])
        ]
        hasil = prediksi_stres(input_data)

        # Tampilkan hasil prediksi
        st.markdown(f"""
            <div style='background-color:{color_mapping[hasil]}; padding:20px; border-radius:10px; text-align:center;'>
                <h2 style='color:white;'>Tingkat Stres: {hasil}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Tampilkan Data Sheet
        st.markdown("### 📄 Tabel Data Google Sheet (5 Terakhir)")
        styled_df = df.tail(5).style.highlight_max(axis=0, color='lightgreen').applymap(
            lambda val: 'background-color: #f9ebae' if val == latest["Suhu (°C)"] or val == latest["SpO2 (%)"] or val == latest["HeartRate (BPM)"] else ''
        )
        st.dataframe(styled_df, use_container_width=True)

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
        st.success(f"Tingkat Stres: {hasil}")
        st.markdown(f"""
            <div style='background-color:{color_mapping[hasil]}; padding:20px; border-radius:10px; text-align:center;'>
                <h2 style='color:white;'>{hasil}</h2>
            </div>
        """, unsafe_allow_html=True)
