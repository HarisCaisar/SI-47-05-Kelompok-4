import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("House Price Prediction Dataset.csv")

# Perbaiki nilai kategori yang typo
data['Location'] = data['Location'].replace({'Sub suburban': 'Suburban'})

# Label Encoding kolom Location
le = LabelEncoder()
data['Location_encoded'] = le.fit_transform(data['Location'])

# Mapping kondisi ke angka
condition_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
data['Condition'] = data['Condition'].map(condition_map)

# Fitur yang dipakai
features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Condition', 'Location_encoded']

# Standarisasi
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Clustering: 3 cluster (Besar, Sedang, Kecil)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Remap cluster ke label berdasarkan rata-rata harga
cluster_avg_price = data.groupby('Cluster')['Price'].mean().sort_values(ascending=False)
cluster_remap = {cluster: label for cluster, label in zip(cluster_avg_price.index, ['Rumah Besar', 'Rumah Sedang', 'Rumah Kecil'])}
data['Cluster_Label'] = data['Cluster'].map(cluster_remap)

# Train model regresi
X = data[features]
y = data['Price']
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X, y)

# Simpan model dan encoder
joblib.dump(regression_model, 'regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans.pkl')
joblib.dump(le, 'location_encoder.pkl')
joblib.dump(cluster_remap, 'cluster_remap.pkl')

print("✅ Model, scaler, encoder, dan label cluster berhasil disimpan.")

# -----------------------------
# 🎨 Visualisasi hasil clustering
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='Area',
    y='Price',
    hue='Cluster_Label',
    palette={'Rumah Besar': 'green', 'Rumah Sedang': 'orange', 'Rumah Kecil': 'red'}
)
plt.title('Visualisasi Cluster Rumah Berdasarkan Area dan Harga')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
