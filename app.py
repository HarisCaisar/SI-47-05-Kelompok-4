import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model dan encoder
model = joblib.load('regression_model.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
le = joblib.load('location_encoder.pkl')
cluster_remap = joblib.load('cluster_remap.pkl')

st.title("🏠 Prediksi Harga Rumah & Klaster Lokasi")

# Input Data Pribadi
st.header("👤 Input Data Pribadi")
nama = st.text_input("Nama Lengkap")
umur = st.number_input("Umur", min_value=18, max_value=100, value=30)
pekerjaan = st.text_input("Pekerjaan")

# Input Data Rumah
st.header("📋 Input Data Rumah")
area = st.number_input("Luas Area (sqft)", min_value=100, max_value=10000, value=2500)
bedrooms = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
bathrooms = st.slider("Jumlah Kamar Mandi", 1, 10, 2)
floors = st.selectbox("Jumlah Lantai", [1, 2, 3])
condition = st.selectbox("Kondisi Rumah", ["Excellent", "Good", "Fair", "Poor"])

# Mapping kondisi ke angka
condition_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
condition_val = condition_map[condition]

# Load data untuk referensi lokasi
data = pd.read_csv("House Price Prediction Dataset.csv")
data['Location'] = data['Location'].replace({'Sub suburban': 'Suburban'})
available_locations = sorted(data['Location'].dropna().unique())
location = st.selectbox("Lokasi (pilih dari data)", options=available_locations)

# Visualisasi pie chart klaster secara keseluruhan
data['Location_encoded'] = le.transform(data['Location'])
data['Condition'] = data['Condition'].map(condition_map)
full_features = data[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Condition', 'Location_encoded']]
scaled_features = scaler.transform(full_features)
data['Cluster'] = kmeans.predict(scaled_features)
data['Cluster_Label'] = data['Cluster'].map(cluster_remap)

cluster_counts = data['Cluster_Label'].value_counts()
labels = cluster_counts.index.tolist()
sizes = cluster_counts.values.tolist()

st.subheader("📊 Distribusi Klaster Rumah")
fig, ax = plt.subplots()
colors = ["#FFC0D9", "#BEADFA", "#C4E2E4"]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Tombol prediksi
if st.button("🔍 Prediksi Harga Rumah"):
    if location not in le.classes_:
        st.error("Lokasi tidak dikenali! Pastikan lokasi valid.")
        st.stop()

    # Encode lokasi
    loc_encoded = le.transform([location])[0]

    # Buat DataFrame input
    input_df = pd.DataFrame([[area, bedrooms, bathrooms, floors, condition_val, loc_encoded]],
                            columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Condition', 'Location_encoded'])

    # Skalakan dan klasterisasi input
    input_scaled = scaler.transform(input_df)
    cluster_pred = kmeans.predict(input_scaled)[0]
    final_cluster = cluster_remap[cluster_pred]

    # Prediksi harga
    price_pred = model.predict(input_df)[0]

    # Output hasil
    st.markdown(f"### 🏷️ Klaster Rumah: **{final_cluster}**")
    st.markdown(f"### 💰 Harga Prediksi: **$ {price_pred:,.0f}**")

    # Skenario naratif
    skenario = f"""
    <h4>Skenario Pembelian Rumah</h4>

    <p><b>Nama:</b> {nama}<br>
    <b>Umur:</b> {umur}<br>
    <b>Pekerjaan:</b> {pekerjaan}</p>

    {nama} sedang mencari rumah baru dan ingin memastikan ia mendapatkan properti yang sesuai dengan preferensinya. Berdasarkan hasil analisis harga dan klasterisasi, berikut adalah beberapa skenario yang bisa ia pertimbangkan.Ia ingin membeli rumah dengan kondisi yang **{condition}** di daerah **{location}**, dengan luas area **{area} sqft**.

    Berdasarkan analisis clustering, rumah yang memenuhi kriteria {nama} adalah **{final_cluster}**. Tipe rumah ini cocok dengan preferensi {nama}.Untuk jumlah kamar tidur, ia ingin rumah dengan **{bedrooms} kamar tidur**, dan juga memilih **{bathrooms} kamar mandi** untuk memenuhi kebutuhannya. Terkait struktur rumah, ia tertarik pada rumah **{floors} lantai** dalam kondisi **{condition}**.
    
    Berdasarkan kriteria tersebut, harga prediksi untuk rumah yang diinginkan {nama} adalah **$ {price_pred:,.0f}**. Ini adalah estimasi harga yang dapat membantu {nama} dalam proses negosiasi dengan penjual.
    """

    st.markdown(skenario, unsafe_allow_html=True)
