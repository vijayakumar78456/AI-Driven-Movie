import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example Movie Data (MovieID, Title, Genres, Description)
movie_data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'The Lord of the Rings: The Return of the King'],
    'genres': ['Drama', 'Crime|Drama', 'Action|Crime|Drama', 'Crime|Drama', 'Action|Adventure|Drama'],
    'description': [
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster’s wife, and a diner’s waitress intertwine in four tales of violence and redemption.',
        'Gandalf and Aragorn lead the World of Men against Sauron’s army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.'
    ]
}

# Example User Data (UserID, MovieID, Rating)
user_ratings = {
    'userId': [1, 1, 1, 2, 2, 3, 3, 3],
    'movieId': [1, 2, 3, 2, 4, 1, 4, 5],
    'rating': [5, 4, 5, 3, 4, 4, 5, 4]
}

# Convert movie and user data into DataFrames
movies_df = pd.DataFrame(movie_data)
ratings_df = pd.DataFrame(user_ratings)

# **Step 1: Collaborative Filtering**
# Create a user-item matrix
user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Function to get movie recommendations based on collaborative filtering
def collaborative_filtering(user_id, num_recommendations=3):
    similar_users = user_similarity[user_id - 1]
    similar_users_indices = similar_users.argsort()[-num_recommendations:][::-1]
    
    recommended_movies = []
    for idx in similar_users_indices:
        user_ratings_for_movie = user_movie_matrix.iloc[idx].to_dict()
        for movie_id, rating in user_ratings_for_movie.items():
            if rating > 0 and movie_id not in recommended_movies:
                recommended_movies.append(movie_id)
    return recommended_movies[:num_recommendations]

# **Step 2: Content-Based Filtering**
# Convert the movie descriptions into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['description'])

# Compute cosine similarity between movies based on descriptions
movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on content similarity
def content_based_filtering(movie_id, num_recommendations=3):
    movie_index = movies_df[movies_df['movieId'] == movie_id].index[0]
    similar_movies = movie_similarity[movie_index]
    similar_movie_indices = similar_movies.argsort()[-num_recommendations-1:-1][::-1]
    
    recommended_movies = [movies_df.iloc[i]['movieId'] for i in similar_movie_indices]
    return recommended_movies

# **Step 3: Hybrid Model**
def hybrid_recommendation(user_id, movie_id, num_recommendations=3):
    collaborative_recs = collaborative_filtering(user_id, num_recommendations)
    content_recs = content_based_filtering(movie_id, num_recommendations)
    
    # Combine the results: prioritize collaborative filtering, then content-based
    combined_recs = list(set(collaborative_recs + content_recs))[:num_recommendations]
    return combined_recs

# Example: Get personalized movie recommendations for User 1
user_id = 1
movie_id = 1  # Example movie id (The Shawshank Redemption)
print(f"Hybrid Recommendations for User {user_id} based on movie {movie_id}:")
print(hybrid_recommendation(user_id, movie_id))
