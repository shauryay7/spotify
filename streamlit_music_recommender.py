"""
Unified Streamlit Music Recommender
Supports multiple dataset types: data.csv, data_by_artist.csv, data_by_genres.csv, data_by_year.csv, data_w_genres.csv

Features:
 - Choose dataset from sidebar (loads from local ./data folder automatically)
 - Auto-detect numeric audio features present and use them for clustering
 - Show metadata columns (artist, track, genres, year) if present
 - Train KMeans, visualize with PCA, show cluster distribution
 - Input custom song features to predict cluster + nearest-neighbor recommendations
 - Download trained model (pickle)

Run:
    streamlit run streamlit_music_recommender.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import io
import pickle
import os

st.set_page_config(page_title="Music Recommender (Unified)", layout="wide")
st.title("🎵 Unified Music Recommender — Local Dataset Mode")

# Sidebar: dataset selection and settings
st.sidebar.header("Dataset & Model Settings")

DATA_PATH = os.path.join(os.getcwd(), "data")

DATASET_OPTIONS = [
    'data.csv',
    'data_by_artist.csv',
    'data_by_genres.csv',
    'data_by_year.csv',
    'data_w_genres.csv'
]

dataset_choice = st.sidebar.selectbox('Choose dataset', DATASET_OPTIONS)
n_clusters = st.sidebar.slider('Number of clusters (K)', min_value=2, max_value=30, value=8)
random_state = int(st.sidebar.number_input('Random seed', value=42, step=1))
train_button = st.sidebar.button('Train / Re-train Model')

# Function to load dataset from ./data
@st.cache_data
def load_local_dataset(name):
    path = os.path.join(DATA_PATH, name)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
            return None
    else:
        st.error(f"File not found: {path}")
        return None

# Load dataset
df = load_local_dataset(dataset_choice)
if df is None:
    st.stop()

st.write(f"### Loaded dataset: `{dataset_choice}`")
st.dataframe(df.head(8))

# Identify possible audio feature columns
COMMON_FEATURES = ['danceability','energy','key','loudness','speechiness','acousticness',
                   'instrumentalness','liveness','valence','tempo','duration_ms']

available_features = [c for c in COMMON_FEATURES if c in df.columns]

# fallback: pick numeric columns if none match
if not available_features:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    remove_like = ['id','track_id','uri']
    numeric_cols = [c for c in numeric_cols if all(r not in c.lower() for r in remove_like)]
    available_features = numeric_cols[:10]

if not available_features:
    st.error('No numeric features detected. Ensure dataset contains valid audio feature columns.')
    st.stop()

st.write(f"Using features: **{', '.join(available_features)}**")

# Detect metadata columns
metadata_cols = []
for name in ['artist_name','track_name','artist','title','genres','genre','year']:
    if name in df.columns:
        metadata_cols.append(name)

if metadata_cols:
    st.write('Detected metadata columns: ' + ', '.join(metadata_cols))

# Preprocessing & training
@st.cache_data
def preprocess_and_train(df_local, features, k, random_state):
    X = df_local[features].fillna(0).astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xs)
    return {'scaler': scaler, 'kmeans': kmeans, 'labels': labels, 'Xs': Xs}

# Train or load session model
if 'model_state' not in st.session_state:
    st.session_state['model_state'] = None

if train_button or st.session_state['model_state'] is None:
    with st.spinner('Training KMeans on selected dataset...'):
        try:
            ms = preprocess_and_train(df, available_features, n_clusters, random_state)
            st.session_state['model_state'] = ms
            df = df.copy()
            df['cluster'] = ms['labels']
            st.success('Model trained successfully.')
        except Exception as e:
            st.error(f'Error training model: {e}')
            st.stop()
else:
    ms = st.session_state['model_state']
    if ms is not None:
        df = df.copy()
        df['cluster'] = ms['labels']

# Cluster distribution
st.write('### Cluster distribution')
cluster_counts = df['cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# PCA visualization
st.write('### 2D PCA visualization of clusters')
try:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(st.session_state['model_state']['Xs'])
    fig, ax = plt.subplots(figsize=(8,5))
    scatter = ax.scatter(coords[:,0], coords[:,1], c=st.session_state['model_state']['labels'], s=20)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title('Songs projected to 2D via PCA')
    st.pyplot(fig)
except Exception as e:
    st.warning(f'PCA plot failed: {e}')

# Predict custom song
st.write('## Predict cluster for a custom song')
with st.form('song_input_form'):
    sliders = {}
    for feat in available_features:
        col = df[feat].dropna()
        lo, hi, med = float(col.min()), float(col.max()), float(col.median())
        if lo >= 0 and hi <= 1:
            val = st.slider(f'{feat}', 0.0, 1.0, med)
        else:
            val = st.number_input(f'{feat}', value=med)
        sliders[feat] = val
    submit = st.form_submit_button('Predict & Recommend')

if submit:
    x = np.array([sliders[f] for f in available_features]).reshape(1, -1)
    scaler = st.session_state['model_state']['scaler']
    kmeans = st.session_state['model_state']['kmeans']
    Xs = st.session_state['model_state']['Xs']
    x_s = scaler.transform(x)
    pred_cluster = int(kmeans.predict(x_s)[0])
    st.success(f'Predicted cluster: {pred_cluster}')

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(Xs)
    dists, idxs = nbrs.kneighbors(x_s)
    recs = df.iloc[idxs[0]]
    display_cols = metadata_cols + available_features
    display_cols = [c for c in display_cols if c in recs.columns]
    st.write('### Top similar songs from dataset')
    st.dataframe(recs[display_cols].reset_index(drop=True))

# Explore clusters
st.write('## Explore songs by cluster')
sel_cluster = st.selectbox('Select cluster', sorted(df['cluster'].unique()))
cluster_df = df[df['cluster'] == sel_cluster]
st.write(f'Showing {len(cluster_df)} songs in cluster {sel_cluster}')
show_cols = metadata_cols + available_features + ['cluster']
show_cols = [c for c in show_cols if c in cluster_df.columns]
st.dataframe(cluster_df[show_cols].head(100))

# Export model
buf = io.BytesIO()
if st.button('Download trained model (pickle)'):
    export_obj = {
        'scaler': st.session_state['model_state']['scaler'],
        'kmeans': st.session_state['model_state']['kmeans'],
        'features': available_features
    }
    pickle.dump(export_obj, buf)
    buf.seek(0)
    st.download_button('Click to download model', data=buf, file_name='music_kmeans_model.pkl')

st.write('---')
st.caption('Unified Streamlit Music Recommender — automatically loads datasets from ./data/')
