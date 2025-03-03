import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle
import os

def train_collaborative():
    # Load ratings data
    ratings = pd.read_csv('dataset/ratings.csv')
    
    # Define reader and load data
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Train model
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/collaborative_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Collaborative model trained and saved successfully!")

def get_collaborative_predictions(user_id, movie_ids):
    # Load the trained model
    with open('models/collaborative_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions
    predictions = []
    for movie_id in movie_ids:
        pred = model.predict(user_id, movie_id)
        predictions.append({'movieId': movie_id, 'predicted_rating': pred.est})
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    train_collaborative()  # Only train the model