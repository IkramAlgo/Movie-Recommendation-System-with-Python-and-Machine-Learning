import pandas as pd
from content_based import get_content_based_recommendations
from collaborative_1 import get_collaborative_predictions  # Updated import

def hybrid_recommendation(user_id, title, top_n=10):
    """
    Generate hybrid recommendations combining content-based and collaborative filtering.
    
    Args:
        user_id (int): ID of the user
        title (str): Title of the movie
        top_n (int): Number of recommendations to return
    
    Returns:
        DataFrame: Top-N hybrid recommendations
    """
    # Content-based recommendations
    content_recs = get_content_based_recommendations(title, top_n)
    
    # Collaborative filtering predictions
    movie_ids = content_recs['movieId'].tolist()
    collaborative_preds = get_collaborative_predictions(user_id, movie_ids)
    
    # Combine results
    hybrid = pd.merge(content_recs, collaborative_preds, on='movieId')
    hybrid['combined_score'] = hybrid['similarity_score'] * 0.5 + hybrid['predicted_rating'] * 0.5
    return hybrid.sort_values('combined_score', ascending=False).head(top_n)

if __name__ == "__main__":
    print(hybrid_recommendation(1, "Toy Story (1995)"))