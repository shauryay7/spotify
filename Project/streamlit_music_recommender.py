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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")

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

# Function to load dataset
@st.cache_data
def load_local_dataset(name):
    path = os.path.join(DATA_PATH, name)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)

            # Fix duplicate column names
            if df.columns.duplicated().any():
                dupes = df.columns[df.columns.duplicated()].tolist()
                st.warning(f"Duplicate columns found and auto-renamed: {dupes}")
                df.columns = pd.io.parsers.base_parser.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

            # Replace infinities and very large values
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            for c in df.select_dtypes(include=[np.number]).columns:
                df[c] = np.clip(df[c], -1e9, 1e9)

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

# Automatically detect numeric feature columns
numeric_cols = []
for col in df.columns:
    try:
        # Convert to numeric and check if most values are numeric
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        numeric_count = numeric_series.notnull().sum()
        total_count = len(df[col].dropna())
        
        # Only include if at least 80% of non-null values are numeric
        if total_count > 0 and numeric_count / total_count >= 0.8:
            # Double-check that the column can actually be used for calculations
            test_min = numeric_series.min()
            if pd.notna(test_min) and (df[col].dtype in [np.float64, np.int64, np.float32, np.int32] or 
                                       numeric_count > 0):
                numeric_cols.append(col)
    except Exception:
        continue

# Filter out obvious non-feature columns (IDs, temporal data, etc.)
remove_like = ['id', 'track_id', 'uri', 'year', 'release_date']
numeric_cols = [c for c in numeric_cols if all(r not in c.lower() for r in remove_like)]

if not numeric_cols:
    st.error('No numeric columns detected. Ensure dataset contains numeric audio features.')
    st.stop()

available_features = numeric_cols
st.write(f"Using numeric features: **{', '.join(available_features)}**")

# Detect metadata columns
metadata_cols = []
# Check for various possible metadata column names
metadata_keywords = ['artist', 'name', 'track', 'title', 'genre', 'year', 'release_date', 'id']
for col in df.columns:
    col_lower = col.lower()
    # Include columns that match metadata keywords (but not if they're numeric features)
    if any(keyword in col_lower for keyword in metadata_keywords) and col not in available_features:
        metadata_cols.append(col)

if metadata_cols:
    st.write('Detected metadata columns: ' + ', '.join(metadata_cols))


# Preprocessing & training
@st.cache_data
def preprocess_and_train(df_local, features, k, random_state):
    X = df_local[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.clip(lower=-1e9, upper=1e9)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Xs)
    return {'scaler': scaler, 'kmeans': kmeans, 'labels': labels, 'Xs': Xs}


# Train or load session model
if 'model_state' not in st.session_state or train_button or st.session_state.get('current_dataset') != dataset_choice:
    with st.spinner('Training KMeans on selected dataset...'):
        try:
            ms = preprocess_and_train(df, available_features, n_clusters, random_state)
            st.session_state['model_state'] = ms
            st.session_state['current_dataset'] = dataset_choice
            st.session_state['trained_features'] = available_features.copy()  # Store features used for training
            df = df.copy()
            df['cluster'] = ms['labels']
            st.success('✅ Model trained successfully.')
        except Exception as e:
            st.error(f'Error training model: {e}')
            st.stop()
else:
    ms = st.session_state['model_state']
    df = df.copy()
    df['cluster'] = ms['labels']

# Use the features that were used during training, not the newly detected ones
trained_features = st.session_state.get('trained_features', available_features)


# Cluster distribution
st.write('### Cluster distribution')
st.bar_chart(df['cluster'].value_counts().sort_index())


# PCA visualization
st.write('### 2D PCA visualization of clusters')
try:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(st.session_state['model_state']['Xs'])
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=st.session_state['model_state']['labels'], s=20)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title('Songs projected to 2D via PCA')
    st.pyplot(fig)
except Exception as e:
    st.warning(f'PCA plot failed: {e}')


# Predict custom song
st.write('## Predict cluster for a custom song')
# Use the features that were used during training
features_for_form = st.session_state.get('trained_features', available_features)
with st.form('song_input_form'):
    sliders = {}
    for feat in features_for_form:
        # Convert to numeric, handling any non-numeric values
        col_numeric = pd.to_numeric(df[feat], errors='coerce').dropna()
        if len(col_numeric) == 0:
            # If column has no valid numeric values, use default
            val = st.number_input(f'{feat}', value=0.0)
        else:
            lo, hi, med = float(col_numeric.min()), float(col_numeric.max()), float(col_numeric.median())
            if np.isfinite(lo) and np.isfinite(hi):
                if lo >= 0 and hi <= 1:
                    val = st.slider(f'{feat}', 0.0, 1.0, float(med))
                else:
                    val = st.number_input(f'{feat}', value=float(med))
            else:
                val = st.number_input(f'{feat}', value=0.0)
        sliders[feat] = val
    submit = st.form_submit_button('Predict & Recommend')


if submit:
    # Use the same features in the same order as training
    x = np.array([sliders[f] for f in features_for_form]).reshape(1, -1)
    scaler = st.session_state['model_state']['scaler']
    kmeans = st.session_state['model_state']['kmeans']
    Xs = st.session_state['model_state']['Xs']
    x_s = scaler.transform(x)
    pred_cluster = int(kmeans.predict(x_s)[0])
    st.success(f'Predicted cluster: {pred_cluster}')

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(Xs)
    dists, idxs = nbrs.kneighbors(x_s)
    recs = df.iloc[idxs[0]]
    display_cols = metadata_cols + features_for_form
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
        'features': st.session_state.get('trained_features', available_features)
    }
    pickle.dump(export_obj, buf)
    buf.seek(0)
    st.download_button('Click to download model', data=buf, file_name='music_kmeans_model.pkl')

st.write('---')
st.caption('Unified Streamlit Music Recommender — automatically loads datasets from ./data/')
