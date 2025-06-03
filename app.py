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

if st.button("Prediksi"):
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

    # Visualisasi pie chart klaster secara keseluruhan
    data['Location'] = data['Location'].replace({'Sub suburban': 'Suburban'})
    data['Location_encoded'] = le.transform(data['Location'])
    data['Condition'] = data['Condition'].map(condition_map)

    full_features = data[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Condition', 'Location_encoded']]
    scaled_features = scaler.transform(full_features)
    data['Cluster'] = kmeans.predict(scaled_features)
    data['Cluster_Label'] = data['Cluster'].map(cluster_remap)

    cluster_counts = data['Cluster_Label'].value_counts()
    labels = cluster_counts.index.tolist()
    sizes = cluster_counts.values.tolist()

    fig, ax = plt.subplots()
    colors = ['#4CAF50', '#FFC107', '#F44336']  # Hijau, Kuning, Merah
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
