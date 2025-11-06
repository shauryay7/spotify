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
st.title("🎵 Music Recommendation System ")

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


# Mood-based recommendations
st.write('---')
st.write('## 🎭 Mood-Based Recommendations')

# Define mood profiles based on audio features
def get_mood_profile(mood_name):
    """Returns feature ranges for different moods"""
    moods = {
        'Happy': {
            'valence': (0.6, 1.0),
            'danceability': (0.6, 1.0),
            'energy': (0.4, 1.0)
        },
        'Sad': {
            'valence': (0.0, 0.4),
            'energy': (0.0, 0.4),
            'danceability': (0.0, 0.5)
        },
        'Energetic': {
            'energy': (0.7, 1.0),
            'tempo': (120, 200),
            'danceability': (0.5, 1.0)
        },
        'Calm/Relaxed': {
            'energy': (0.0, 0.4),
            'tempo': (60, 100),
            'acousticness': (0.5, 1.0),
            'valence': (0.3, 0.7)
        },
        'Romantic': {
            'valence': (0.4, 0.6),
            'energy': (0.0, 0.5),
            'tempo': (60, 100),
            'acousticness': (0.3, 1.0)
        },
        'Party': {
            'danceability': (0.7, 1.0),
            'energy': (0.7, 1.0),
            'tempo': (120, 200),
            'valence': (0.5, 1.0)
        },
        'Focused/Study': {
            'instrumentalness': (0.3, 1.0),
            'energy': (0.2, 0.6),
            'speechiness': (0.0, 0.3),
            'tempo': (70, 120)
        },
        'Workout': {
            'energy': (0.8, 1.0),
            'tempo': (130, 200),
            'danceability': (0.6, 1.0),
            'valence': (0.5, 1.0)
        }
    }
    return moods.get(mood_name, {})

# Mood selection
mood_options = ['Happy', 'Sad', 'Energetic', 'Calm/Relaxed', 'Romantic', 'Party', 'Focused/Study', 'Workout']
selected_mood = st.selectbox('Select your mood', mood_options)

num_recommendations = st.slider('Number of recommendations', min_value=5, max_value=50, value=20)

if st.button('Get Mood-Based Recommendations'):
    mood_profile = get_mood_profile(selected_mood)

    if not mood_profile:
        st.error('Invalid mood selected')
    else:
        # Start with all songs
        mood_df = df.copy()

        # Filter songs based on mood profile
        for feature, (min_val, max_val) in mood_profile.items():
            if feature in mood_df.columns:
                # Convert to numeric
                mood_df[feature] = pd.to_numeric(mood_df[feature], errors='coerce')
                # Filter by range
                mood_df = mood_df[
                    (mood_df[feature] >= min_val) &
                    (mood_df[feature] <= max_val)
                    ]
            else:
                # If feature doesn't exist, show warning but continue
                st.warning(f"Feature '{feature}' not found in dataset. Skipping this filter.")

        if len(mood_df) == 0:
            st.warning(f'No songs found matching the "{selected_mood}" mood profile. Try a different mood or adjust the filters.')
        else:
            # Score songs based on how well they match the mood
            # Calculate a match score (sum of normalized feature values)
            score_data = {}
            for feature in mood_profile.keys():
                if feature in mood_df.columns:
                    mood_df[feature] = pd.to_numeric(mood_df[feature], errors='coerce')
                    # Normalize feature to 0-1 range for scoring
                    feature_min = mood_df[feature].min()
                    feature_max = mood_df[feature].max()
                    if feature_max > feature_min:
                        normalized = (mood_df[feature] - feature_min) / (feature_max - feature_min)
                    else:
                        normalized = mood_df[feature] / feature_max if feature_max > 0 else mood_df[feature]
                    score_data[feature] = normalized

            if score_data:
                # Create a DataFrame from normalized scores and average them
                score_df = pd.DataFrame(score_data)
                mood_df['mood_match_score'] = score_df.mean(axis=1)
                # Sort by match score (descending)
                mood_df = mood_df.sort_values('mood_match_score', ascending=False)
            else:
                # If no scoring possible, just shuffle
                mood_df = mood_df.sample(frac=1).reset_index(drop=True)

            # Display top recommendations
            st.success(f'Found {len(mood_df)} songs matching "{selected_mood}" mood!')
            recommendations = mood_df.head(num_recommendations)

            # Prepare display columns
            display_cols_mood = metadata_cols.copy()
            # Add mood-relevant features
            for feat in mood_profile.keys():
                if feat in recommendations.columns and feat not in display_cols_mood:
                    display_cols_mood.append(feat)
            # Add match score if available
            if 'mood_match_score' in recommendations.columns:
                display_cols_mood.append('mood_match_score')

            display_cols_mood = [c for c in display_cols_mood if c in recommendations.columns]

            st.write(f'### Top {min(num_recommendations, len(recommendations))} Recommendations for "{selected_mood}" Mood')
            st.dataframe(recommendations[display_cols_mood].reset_index(drop=True))

            # Show mood profile info
            with st.expander('View Mood Profile Details'):
                st.write(f'**{selected_mood} Mood Characteristics:**')
                for feature, (min_val, max_val) in mood_profile.items():
                    if feature in df.columns:
                        st.write(f'- **{feature}**: {min_val:.2f} - {max_val:.2f}')


# Year-based recommendations
st.write('---')
st.write('## 📅 Year-Based Recommendations')

# Check if 'year' column exists in the dataset
year_col = None
for col in ['year', 'Year', 'YEAR']:
    if col in df.columns:
        year_col = col
        break

if year_col:
    # Get year range from dataset
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    valid_years = df[year_col].dropna()

    if len(valid_years) > 0:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())

        st.write(f'Available years: {min_year} - {max_year}')

        # Year selection options
        year_option = st.radio(
            'Select year option',
            ['Single Year', 'Year Range'],
            horizontal=True
        )

        year_filter_df = pd.DataFrame()

        if year_option == 'Single Year':
            selected_year = st.slider('Select year', min_value=min_year, max_value=max_year, value=max_year)
            year_filter_df = df[df[year_col] == selected_year].copy()
            st.write(f'**Filtering songs from {selected_year}**')
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.slider('Start year', min_value=min_year, max_value=max_year, value=min_year)
            with col2:
                end_year = st.slider('End year', min_value=min_year, max_value=max_year, value=max_year)

            if start_year > end_year:
                st.error('Start year must be less than or equal to end year')
            else:
                year_filter_df = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)].copy()
                st.write(f'**Filtering songs from {start_year} to {end_year}**')

        if len(year_filter_df) > 0:
            # Sorting options
            sort_by = st.selectbox(
                'Sort by',
                ['Popularity (High to Low)', 'Popularity (Low to High)',
                 'Year (Newest First)', 'Year (Oldest First)',
                 'Energy (High to Low)', 'Danceability (High to Low)',
                 'Valence (High to Low)', 'Tempo (High to Low)',
                 'Random']
            )

            num_recommendations_year = st.slider(
                'Number of recommendations',
                min_value=5,
                max_value=100,
                value=20,
                key='year_recs'
            )

            if st.button('Get Year-Based Recommendations'):
                # Apply sorting
                if sort_by == 'Popularity (High to Low)':
                    if 'popularity' in year_filter_df.columns:
                        year_filter_df['popularity'] = pd.to_numeric(year_filter_df['popularity'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('popularity', ascending=False, na_position='last')
                    else:
                        st.warning("'Popularity' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                elif sort_by == 'Popularity (Low to High)':
                    if 'popularity' in year_filter_df.columns:
                        year_filter_df['popularity'] = pd.to_numeric(year_filter_df['popularity'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('popularity', ascending=True, na_position='last')
                    else:
                        st.warning("'Popularity' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=True)
                elif sort_by == 'Year (Newest First)':
                    year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                elif sort_by == 'Year (Oldest First)':
                    year_filter_df = year_filter_df.sort_values(year_col, ascending=True)
                elif sort_by == 'Energy (High to Low)':
                    if 'energy' in year_filter_df.columns:
                        year_filter_df['energy'] = pd.to_numeric(year_filter_df['energy'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('energy', ascending=False, na_position='last')
                    else:
                        st.warning("'Energy' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                elif sort_by == 'Danceability (High to Low)':
                    if 'danceability' in year_filter_df.columns:
                        year_filter_df['danceability'] = pd.to_numeric(year_filter_df['danceability'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('danceability', ascending=False, na_position='last')
                    else:
                        st.warning("'Danceability' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                elif sort_by == 'Valence (High to Low)':
                    if 'valence' in year_filter_df.columns:
                        year_filter_df['valence'] = pd.to_numeric(year_filter_df['valence'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('valence', ascending=False, na_position='last')
                    else:
                        st.warning("'Valence' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                elif sort_by == 'Tempo (High to Low)':
                    if 'tempo' in year_filter_df.columns:
                        year_filter_df['tempo'] = pd.to_numeric(year_filter_df['tempo'], errors='coerce')
                        year_filter_df = year_filter_df.sort_values('tempo', ascending=False, na_position='last')
                    else:
                        st.warning("'Tempo' column not found. Sorting by year instead.")
                        year_filter_df = year_filter_df.sort_values(year_col, ascending=False)
                else:  # Random
                    year_filter_df = year_filter_df.sample(frac=1).reset_index(drop=True)

                # Get recommendations
                recommendations_year = year_filter_df.head(num_recommendations_year)

                # Prepare display columns
                display_cols_year = metadata_cols.copy()
                if year_col not in display_cols_year:
                    display_cols_year.append(year_col)

                # Add sorting column if it's a feature
                sort_col_map = {
                    'Popularity (High to Low)': 'popularity',
                    'Popularity (Low to High)': 'popularity',
                    'Energy (High to Low)': 'energy',
                    'Danceability (High to Low)': 'danceability',
                    'Valence (High to Low)': 'valence',
                    'Tempo (High to Low)': 'tempo'
                }

                if sort_by in sort_col_map:
                    sort_col = sort_col_map[sort_by]
                    if sort_col in recommendations_year.columns and sort_col not in display_cols_year:
                        display_cols_year.append(sort_col)

                # Add some common audio features
                for feat in ['energy', 'danceability', 'valence', 'tempo', 'popularity']:
                    if feat in recommendations_year.columns and feat not in display_cols_year:
                        display_cols_year.append(feat)

                display_cols_year = [c for c in display_cols_year if c in recommendations_year.columns]

                st.success(f'Found {len(year_filter_df)} songs in selected year(s)!')
                st.write(f'### Top {min(num_recommendations_year, len(recommendations_year))} Recommendations ({sort_by})')
                st.dataframe(recommendations_year[display_cols_year].reset_index(drop=True))

                # Show year distribution
                if year_option == 'Year Range':
                    with st.expander('View Year Distribution'):
                        year_counts = year_filter_df[year_col].value_counts().sort_index()
                        st.bar_chart(year_counts)
        else:
            st.info(f'No songs found in the selected year(s). Try selecting a different year or year range.')
    else:
        st.warning("No valid year data found in the dataset.")
else:
    st.info("⚠️ 'Year' column not found in this dataset. Year-based recommendations are not available.")
    st.write("Available columns:", ', '.join(df.columns.tolist()[:10]))


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
