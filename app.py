import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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
data = pd.read_csv("House Price Prediction Dataset.csv")
available_locations = sorted(data['Location'].unique())
location = st.selectbox("Lokasi (pilih dari data)", options=available_locations)


if st.button("Prediksi"):
    loc_encoded = le.transform([location])[0] if location in le.classes_ else 0

    input_data = pd.DataFrame([[area, bedrooms, bathrooms, floors, loc_encoded]],
                              columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Location_encoded'])

    scaled_input = scaler.transform(input_data)
    raw_cluster = kmeans.predict(scaled_input)[0]
    final_cluster = cluster_remap[raw_cluster]
    kategori = ['Rumah kecil', 'Rumah sedang', 'Rumah besar'][final_cluster]

    input_data_for_price = input_data.copy()
    harga = model.predict(input_data_for_price)[0]

    st.markdown(f"### 🏷️ Klaster Rumah: **{kategori}**")
    st.markdown(f"### 💰 Harga Prediksi: **$ {harga:,.0f}**")

    st.subheader("📊 Distribusi Klaster Rumah (Pie Chart)")
    data = pd.read_csv("House Price Prediction Dataset.csv")
    data['Location_encoded'] = le.transform(data['Location'])
    scaled = scaler.transform(data[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Location_encoded']])
    data['Cluster'] = kmeans.predict(scaled)
    data['Cluster'] = data['Cluster'].map(cluster_remap)

    cluster_counts = data['Cluster'].value_counts().sort_index()
    labels = ['Rumah kecil', 'Rumah sedang', 'Rumah besar']
    sizes = [cluster_counts.get(i, 0) for i in range(3)]

    fig, ax = plt.subplots()
    colors = ['#ff9999','#66b3ff','#99ff99']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
