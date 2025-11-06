# 🎵 Music Recommendation System Using Spotify Dataset

A comprehensive music recommendation system that addresses the limitations of traditional recommendation systems by providing personalized, diverse, and scalable music suggestions using hybrid recommendation approaches.

## 📋 Problem Statement

The current music recommendation system is hindered by its inability to provide personalized, diverse, and scalable music suggestions. This is primarily because it:

- **Over-relies on song popularity** - limiting discovery of niche music
- **Uses simplistic audio features** - ignoring critical factors like mood, lyrics, and personal taste
- **Struggles with cold start problem** - for new users/songs
- **Faces scalability issues** - when clustering large datasets

## ✨ Solution Features

This project addresses all these issues through:

### 1. **Hybrid Recommendation System**
   - **Content-Based Filtering**: Recommends songs based on audio feature similarity
   - **Collaborative Filtering**: Uses popularity and cluster-based approaches
   - **Hybrid Scoring**: Weighted combination of both approaches for balanced recommendations

### 2. **Cold Start Handling**
   - **New Users**: Uses content-based recommendations (no history required)
   - **New Songs**: Uses collaborative filtering with diversity (popular songs from various clusters)

### 3. **Scalability Optimizations**
   - **MiniBatch KMeans**: 3-10x faster clustering for large datasets
   - **Dataset Sampling**: Optional sampling for very large datasets
   - **Efficient Processing**: Handles datasets with 100K+ songs

### 4. **Mood-Based Recommendations**
   - 8 different mood profiles (Happy, Sad, Energetic, Calm/Relaxed, Romantic, Party, Focused/Study, Workout)
   - Audio feature-based mood matching
   - Customizable number of recommendations

### 5. **Year-Based Recommendations**
   - Single year or year range selection
   - Multiple sorting options (popularity, energy, danceability, etc.)
   - Year distribution visualization

### 6. **Advanced Features**
   - Automatic feature detection from multiple dataset types
   - PCA visualization of clusters
   - Model export functionality
   - Support for multiple dataset formats

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd /path/to/Project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install required packages**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib
   ```

   Or install all at once:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib
   ```

5. **Ensure your data folder structure**
   ```
   Project/
   ├── data/
   │   ├── data.csv
   │   ├── data_by_artist.csv
   │   ├── data_by_genres.csv
   │   ├── data_by_year.csv
   │   └── data_w_genres.csv
   ├── streamlit_music_recommender.py
   └── README.md
   ```

## 📖 Usage

### Starting the Application

1. **Activate your virtual environment** (if not already active)
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_music_recommender.py
   ```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Or navigate manually to the URL shown in the terminal

### Using the Application

#### **Step 1: Configure Dataset & Model Settings (Sidebar)**

1. **Select Dataset**: Choose from available CSV files in the `data/` folder
2. **Set Clusters**: Adjust number of clusters (K) for KMeans (default: 8)
3. **Random Seed**: Set for reproducibility (default: 42)
4. **Scalability Options**:
   - ✅ Enable "MiniBatch KMeans" for faster processing (recommended for large datasets)
   - Optionally enable "Use sampling" and set sample size for very large datasets
5. **Click "Train / Re-train Model"** to train the clustering model

#### **Step 2: Hybrid Recommendations**

1. **Set Recommendation Options**:
   - Check "New User (Cold Start)" if this is a new user
   - Check "New Song (Cold Start)" if recommending for a new song
   - Adjust content-based and collaborative weights (default: 0.6 and 0.4)
   - Set number of recommendations

2. **Input Song Features**:
   - Fill in audio feature values for the song you want recommendations for
   - Features are automatically detected from your dataset
   - Sliders are provided for features in 0-1 range
   - Number inputs for other numeric features

3. **Click "Predict & Recommend"**:
   - View predicted cluster
   - See hybrid recommendations with scores
   - Check recommendation breakdown (content-based vs collaborative)

#### **Step 3: Mood-Based Recommendations**

1. **Select a Mood**: Choose from 8 available moods
2. **Set Number of Recommendations**: Adjust slider (5-50)
3. **Click "Get Mood-Based Recommendations"**:
   - View songs matching your selected mood
   - See mood match scores
   - Expand "View Mood Profile Details" to see feature ranges

#### **Step 4: Year-Based Recommendations**

1. **Select Year Option**: Single year or year range
2. **Choose Sorting**: Popularity, Year, Energy, Danceability, Valence, Tempo, or Random
3. **Set Number of Recommendations**: Adjust slider (5-100)
4. **Click "Get Year-Based Recommendations"**:
   - View recommendations sorted by your preference
   - For year ranges, see year distribution chart

#### **Step 5: Explore Clusters**

- Select a cluster from the dropdown
- View songs in that cluster
- See metadata and audio features

#### **Step 6: Export Model**

- Click "Download trained model (pickle)" to save the trained model
- Model includes scaler, KMeans model, and feature names

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                            │
│  (Song Features / Mood / Year / Custom Preferences)     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Preprocessing                          │
│  • Load CSV Dataset                                      │
│  • Auto-detect numeric features                          │
│  • Handle missing values & outliers                      │
│  • StandardScaler normalization                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Clustering (KMeans / MiniBatch KMeans)      │
│  • Train on audio features                               │
│  • Generate clusters                                     │
│  • PCA visualization                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Hybrid Recommendation Engine                     │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Content-Based    │  │ Collaborative    │           │
│  │ (Similarity)     │  │ (Popularity +    │           │
│  │                  │  │  Cluster-based) │           │
│  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                      │
│           └──────────┬───────────┘                      │
│                     ▼                                   │
│            Weighted Combination                         │
│            (Hybrid Score)                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Recommendations Output                      │
│  • Ranked by hybrid score                               │
│  • Includes metadata & audio features                   │
│  • Handles cold start cases                             │
└─────────────────────────────────────────────────────────┘
```

### Key Algorithms & Techniques

#### 1. **KMeans Clustering**
- **Purpose**: Group similar songs based on audio features
- **Algorithm**: 
  - Standard KMeans: Full dataset processing, more accurate
  - MiniBatch KMeans: Processes in batches, 3-10x faster
- **Process**:
  1. Feature extraction from audio data
  2. StandardScaler normalization
  3. KMeans clustering (K clusters)
  4. Assign cluster labels to songs

#### 2. **Content-Based Filtering**
- **Method**: Nearest Neighbors algorithm
- **Process**:
  1. Calculate Euclidean distance between input song and all songs
  2. Find K nearest neighbors
  3. Score by inverse distance (closer = higher score)
  4. Return top N recommendations

#### 3. **Collaborative Filtering**
- **Method**: Cluster-based + Popularity scoring
- **Process**:
  1. Predict cluster for input song
  2. Get songs from same cluster
  3. Score by popularity (normalized)
  4. Return top N recommendations

#### 4. **Hybrid Recommendation**
- **Formula**: 
  ```
  Hybrid Score = (α × Content Score) + (β × Collaborative Score)
  ```
  Where:
  - α = Content-based weight (default: 0.6)
  - β = Collaborative weight (default: 0.4)
  - α + β = 1.0

#### 5. **Cold Start Handling**

**New User (Cold Start)**:
- Problem: No user history available
- Solution: Use content-based filtering only
- Process: Find similar songs based on audio features
- Advantage: Works immediately without user data

**New Song (Cold Start)**:
- Problem: New song not in dataset
- Solution: Use collaborative filtering with diversity
- Process: 
  1. Predict cluster for new song
  2. Get popular songs from same cluster
  3. Ensure diversity across clusters
- Advantage: Recommends popular songs that match the new song's characteristics

#### 6. **Mood-Based Recommendations**
- **Method**: Feature range filtering + scoring
- **Process**:
  1. Define mood profiles (feature ranges for each mood)
  2. Filter songs matching mood criteria
  3. Calculate match score (normalized feature values)
  4. Sort by match score
  5. Return top N recommendations

**Mood Profiles** (example):
- **Happy**: valence > 0.6, danceability > 0.6, energy > 0.4
- **Sad**: valence < 0.4, energy < 0.4, danceability < 0.5
- **Energetic**: energy > 0.7, tempo 120-200, danceability > 0.5

#### 7. **Year-Based Recommendations**
- **Method**: Temporal filtering + multiple sorting options
- **Process**:
  1. Filter songs by selected year(s)
  2. Apply sorting (popularity, energy, danceability, etc.)
  3. Return top N recommendations
  4. Show year distribution (for ranges)

### Feature Detection

The system automatically detects:
- **Numeric Features**: Audio features (valence, energy, danceability, tempo, etc.)
  - Requires 80%+ numeric values
  - Excludes IDs, temporal data (year, release_date)
  
- **Metadata Columns**: Artist names, track names, genres, etc.
  - Detected by keyword matching
  - Excluded from clustering features

### Scalability Features

1. **MiniBatch KMeans**:
   - Processes data in batches (batch_size=256)
   - Reduces memory usage
   - 3-10x faster than standard KMeans
   - Suitable for datasets with 10K+ songs

2. **Dataset Sampling**:
   - Optional random sampling for very large datasets
   - Maintains statistical properties
   - Reduces processing time
   - Recommended for datasets with 100K+ songs

## 📊 Dataset Format

### Required CSV Structure

Your dataset should contain columns for:
- **Audio Features**: numeric values (valence, energy, danceability, tempo, acousticness, etc.)
- **Metadata** (optional): artist names, track names, genres, year, popularity

### Supported Datasets

The system automatically works with:
- `data.csv` - Main dataset
- `data_by_artist.csv` - Songs grouped by artist
- `data_by_genres.csv` - Songs grouped by genre
- `data_by_year.csv` - Songs grouped by year
- `data_w_genres.csv` - Songs with genre information

### Example Dataset Format

```csv
valence,year,acousticness,artists,danceability,duration_ms,energy,explicit,id,instrumentalness,key,liveness,loudness,mode,name,popularity,release_date,speechiness,tempo
0.0594,1921,0.982,['Artist Name'],0.279,831667,0.211,0,4BJqT0PrAfrxzMOxytFOIz,0.878,10,0.665,-20.096,1,Song Name,5,1921,0.415,60.936
```

## 🎯 Use Cases

1. **Music Discovery**: Find new songs similar to your favorites
2. **Playlist Creation**: Generate playlists based on mood or year
3. **Niche Music Discovery**: Discover less popular songs that match your taste
4. **Personalized Recommendations**: Get recommendations tailored to your preferences
5. **Cold Start Scenarios**: Handle new users or new songs effectively

## 🔍 Technical Details

### Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
  - KMeans, MiniBatchKMeans
  - StandardScaler
  - PCA
  - NearestNeighbors
- **matplotlib**: Data visualization

### Performance

- **Training Time**: 
  - Standard KMeans: ~10-30 seconds for 10K songs
  - MiniBatch KMeans: ~3-10 seconds for 10K songs
- **Recommendation Time**: < 1 second per query
- **Memory Usage**: Scales with dataset size (sampling helps)

### Limitations

1. **No Lyrics Analysis**: Currently only uses audio features
2. **No Real User Profiles**: Simulates user preferences via popularity
3. **Local Processing**: Runs on local machine (not cloud-scalable)
4. **Single Machine Learning**: No ensemble methods

## 🚧 Future Enhancements

Potential improvements:
- [ ] Lyrics sentiment analysis
- [ ] Real user profile tracking
- [ ] Deep learning models (neural collaborative filtering)
- [ ] Real-time model updates
- [ ] Multi-user support
- [ ] API endpoints
- [ ] Cloud deployment

## 📝 Notes

- The system uses caching (`@st.cache_data`) for efficient data loading
- Models are stored in session state for persistence during the session
- All features are automatically normalized using StandardScaler
- The system handles missing values and outliers automatically

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Spotify for providing audio feature datasets
- scikit-learn for machine learning algorithms
- Streamlit for the web framework

---

**Built with ❤️ using Python, Streamlit, and scikit-learn**

