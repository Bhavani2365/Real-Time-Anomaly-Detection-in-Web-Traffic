# Streamlit Web App: Anomaly Detection in Web Traffic Logs (No Training Data Required)

# Streamlit App: Advanced Web Log Anomaly Detector (No Training Data)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import entropy
from datetime import datetime
import io

st.set_page_config(page_title="Advanced Web Log Anomaly Detector", layout="wide")

# -------------------------------
# Data Generation & Upload
# -------------------------------
def generate_fake_log_data(n=1000):
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=n, freq='min')
    ips = [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n)]
    urls = [f"/page/{np.random.randint(1,10)}" for _ in range(n)]
    codes = np.random.choice([200, 404, 500, 403], size=n, p=[0.85, 0.1, 0.03, 0.02])
    bytes_sent = np.random.randint(100, 5000, size=n)
    user_agents = np.random.choice(["Mozilla/5.0", "Chrome/90.0", "BotCrawler/1.0"], p=[0.6, 0.3, 0.1], size=n)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'ip': ips,
        'url': urls,
        'status_code': codes,
        'bytes': bytes_sent,
        'user_agent': user_agents
    })
    return df

# -------------------------------
# Advanced Feature Engineering
# -------------------------------
def add_session_features(df):
    df_sorted = df.sort_values(by=['ip', 'timestamp'])
    session_info = df_sorted.groupby('ip').agg({
        'timestamp': [lambda x: (x.max() - x.min()).total_seconds(), 'count']
    })
    session_info.columns = ['session_duration', 'request_count']
    session_info['requests_per_min'] = session_info['request_count'] / (session_info['session_duration'] / 60 + 1e-5)
    return session_info.reset_index()

def compute_path_entropy(df):
    def shannon_entropy(urls):
        vals, counts = np.unique(urls, return_counts=True)
        return entropy(counts, base=2)
    entropies = df.groupby('ip')['url'].apply(shannon_entropy).reset_index()
    entropies.columns = ['ip', 'path_entropy']
    return entropies

def status_code_ratios(df):
    return df.groupby('ip')['status_code'].value_counts(normalize=True).unstack(fill_value=0).reset_index()

def advanced_engineer_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['is_bot'] = df['user_agent'].str.contains("bot|crawler", case=False).astype(int)

    basic = df.groupby('ip').agg({
        'url': 'nunique',
        'status_code': lambda x: (x == 404).mean(),
        'bytes': ['mean', 'sum'],
        'is_bot': 'mean',
        'hour': lambda x: x.mode()[0] if len(x) > 0 else -1
    })
    basic.columns = ['url_unique', '404_rate', 'bytes_mean', 'bytes_sum', 'bot_rate', 'common_hour']
    basic = basic.reset_index()

    session = add_session_features(df)
    ua = df.groupby('ip')['user_agent'].nunique().reset_index().rename(columns={'user_agent': 'unique_user_agents'})
    entropy_df = compute_path_entropy(df)
    status_ratios = status_code_ratios(df)

    merged = basic.merge(session, on='ip', how='left')\
                  .merge(ua, on='ip', how='left')\
                  .merge(entropy_df, on='ip', how='left')\
                  .merge(status_ratios, on='ip', how='left')
    return merged

# -------------------------------
# Anomaly Detection Methods
# -------------------------------
def run_dbscan(data):
    model = DBSCAN(eps=0.9, min_samples=3)
    return model.fit_predict(data)

def run_isolation_forest(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    return model.fit_predict(data)

def run_lof(data):
    model = LocalOutlierFactor(n_neighbors=20)
    return model.fit_predict(data)

# -------------------------------
# Visualize Clusters
# -------------------------------
def visualize_clusters(pca_data, labels, method):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=labels, palette='tab10', ax=ax)
    ax.set_title(f"Anomaly Detection using {method}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True)
    return fig

# -------------------------------
# Streamlit Layout
# -------------------------------
st.title("🔍 Advanced Web Traffic Log Anomaly Detector")
st.markdown("Detect abnormal IP behavior in web logs using unsupervised learning (no training data).")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📄 Input Data")
    method = st.radio("Select data source:", ["Generate Synthetic Logs", "Upload Log CSV"], horizontal=True)

    if method == "Generate Synthetic Logs":
        n_rows = st.slider("Number of log entries to generate:", 500, 5000, 1000, 100)
        df_logs = generate_fake_log_data(n_rows)
    else:
        uploaded_file = st.file_uploader("Upload a CSV log file", type=["csv"])
        if uploaded_file:
            df_logs = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        else:
            st.warning("Please upload a log file with at least 'timestamp', 'ip', 'url', 'status_code', 'bytes', and 'user_agent'.")
            st.stop()

    st.write("### Sample Logs")
    st.dataframe(df_logs.head(), use_container_width=True)

with col2:
    st.subheader("⚙️ Feature Engineering & Detection")
    features_df = advanced_engineer_features(df_logs)

    st.write("### Advanced Engineered Features")
    st.dataframe(features_df.head(), use_container_width=True)

    # Normalize and reduce dimensions
    X = features_df.drop('ip', axis=1).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    algo = st.selectbox("Select Detection Algorithm:", ["DBSCAN", "Isolation Forest", "Local Outlier Factor"])
    if algo == "DBSCAN":
        labels = run_dbscan(pca_data)
    elif algo == "Isolation Forest":
        labels = run_isolation_forest(X_scaled)
    else:
        labels = run_lof(X_scaled)

    features_df['anomaly'] = labels

    st.write("### Anomalous IPs")
    if algo == "DBSCAN":
        anomalies = features_df[features_df['anomaly'] == -1]
    else:
        anomalies = features_df[features_df['anomaly'] == -1]

    st.dataframe(anomalies if not anomalies.empty else "No anomalies detected.", use_container_width=True)

st.subheader("📊 Visualization")
fig = visualize_clusters(pca_data, labels, algo)
st.pyplot(fig)

csv = features_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Results as CSV", csv, "advanced_anomaly_detection.csv", "text/csv")
