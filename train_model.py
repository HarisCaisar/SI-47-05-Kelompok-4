import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("House Price Prediction Dataset.csv")

le = LabelEncoder()
data['Location_encoded'] = le.fit_transform(data['Location'])

features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Location_encoded']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

avg_area = data.groupby('Cluster')['Area'].mean()
cluster_order = avg_area.sort_values().index.tolist()
cluster_remap = {old: new for new, old in enumerate(cluster_order)}
data['Cluster'] = data['Cluster'].map(cluster_remap)

X = data[features]  
y = data['Price']
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X, y)


joblib.dump(regression_model, 'regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  
joblib.dump(kmeans, 'kmeans.pkl')
joblib.dump(le, 'location_encoder.pkl')
joblib.dump(cluster_remap, 'cluster_remap.pkl')

print("Model dan encoder disimpan dengan benar!")
