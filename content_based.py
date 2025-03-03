import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def train_content_based():
    """
    Train the content-based model using movie genres and save the similarity matrix.
    """
    # Load dataset
    movies = pd.read_csv('dataset/movies.csv', low_memory=False)
    
    # Handle missing genres
    if 'genres' not in movies.columns:
        raise ValueError("The 'genres' column is missing in movies.csv!")
    movies['genres'] = movies['genres'].fillna("")
    
    # Create TF-IDF matrix for genres
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies['genres'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save model and similarity matrix
    os.makedirs('models', exist_ok=True)
    with open('models/content_similarity.pkl', 'wb') as f:
        pickle.dump((vectorizer, cosine_sim), f)
    
    print("Content-based model trained and saved successfully!")

def get_content_based_recommendations(title, top_n=10):
    """
    Get content-based recommendations for a given movie title.
    
    Args:
        title (str): Title of the movie
        top_n (int): Number of recommendations to return
    
    Returns:
        DataFrame: Top-N recommended movies with titles, genres, movieId, and similarity_score
    """
    # Load similarity matrix and movies data
    with open('models/content_similarity.pkl', 'rb') as f:
        vectorizer, cosine_sim = pickle.load(f)
    
    movies = pd.read_csv('dataset/movies.csv')
    
    # Get movie index
    try:
        idx = movies.index[movies['title'] == title].tolist()[0]
    except IndexError:
        return f"Movie '{title}' not found in the dataset."
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude self
    
    # Get movie indices and similarity scores
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Create DataFrame with movieId, title, genres, and similarity_score
    recommendations = movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
    recommendations['similarity_score'] = similarity_scores
    
    return recommendations

if __name__ == "__main__":
    # Train the model (if not already trained)
    train_content_based()
    
    # Test recommendations
    test_title = "Toy Story (1995)"
    print(f"Recommendations for '{test_title}':")
    print(get_content_based_recommendations(test_title))