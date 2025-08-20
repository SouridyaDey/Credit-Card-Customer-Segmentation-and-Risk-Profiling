from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load pipeline components
imputer = pickle.load(open("knn_imputer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))

# Cluster label mapping
cluster_labels = {
    0: "Moderate Risk / Balanced Users",
    1: "Low Risk / Transactors",
    2: "High Risk / Revolvers"
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    input_features = [
        'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
        'CASH_ADVANCE_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','PAYMENTS', 'MINIMUM_PAYMENTS'
        'CREDIT_LIMIT', 'PRC_FULL_PAYMENT', 'TENURE'
    ]
    
    data = [request.form.get(feat) for feat in input_features]

    # Convert to DataFrame
    df = pd.DataFrame([data], columns=input_features)
    df = df.astype(float)  # Ensure numeric

    # Impute missing values
    df_imputed = imputer.transform(df)

    # Scale
    df_scaled = scaler.transform(df_imputed)

    # PCA
    df_pca = pca.transform(df_scaled)

    # Predict cluster
    cluster = kmeans.predict(df_pca)[0]

    # Map to label
    label = cluster_labels.get(cluster, "Unknown")

    return render_template("index.html", prediction_text=f"Cluster: {cluster} → {label}")

if __name__ == '__main__':
    app.run(debug=True)
