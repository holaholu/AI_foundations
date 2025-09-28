#import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample movie ratings data (since we don't have the actual dataset)
np.random.seed(42)

# Generate sample data
users = range(1, 101)  # 100 users
movies = range(1, 51)  # 50 movies
movie_titles = [f"Movie_{i}" for i in movies]

# Create ratings data
ratings_data = []
for user in users:
    # Each user rates 10-30 random movies
    num_ratings = np.random.randint(10, 31)
    user_movies = np.random.choice(movies, num_ratings, replace=False)
    for movie in user_movies:
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
        ratings_data.append({'userId': user, 'movieId': movie, 'rating': rating})

ratings = pd.DataFrame(ratings_data)
print("Sample ratings data:")
print(ratings.head())
print(f"\nDataset shape: {ratings.shape}")
print(f"Number of users: {ratings['userId'].nunique()}")
print(f"Number of movies: {ratings['movieId'].nunique()}")

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print(f"\nUser-Movie Matrix shape: {user_movie_matrix.shape}")

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Create training matrix
train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Matrix Factorization using Truncated SVD
class MovieRecommendationSystem:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = None
        
    def fit(self, user_movie_matrix):
        # Calculate global mean
        self.global_mean = user_movie_matrix[user_movie_matrix > 0].mean().mean()
        
        # Center the data
        centered_matrix = user_movie_matrix.copy()
        centered_matrix[centered_matrix > 0] -= self.global_mean
        
        # Apply SVD
        self.user_factors = self.svd.fit_transform(centered_matrix)
        self.movie_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_id, movie_id):
        if user_id not in range(len(self.user_factors)) or movie_id not in range(len(self.movie_factors)):
            return self.global_mean
        
        prediction = np.dot(self.user_factors[user_id], self.movie_factors[movie_id]) + self.global_mean
        return max(1, min(5, prediction))  # Clip to rating range
    
    def predict_all(self, test_data):
        predictions = []
        for _, row in test_data.iterrows():
            user_idx = row['userId'] - 1  # Convert to 0-based index
            movie_idx = row['movieId'] - 1  # Convert to 0-based index
            pred = self.predict(user_idx, movie_idx)
            predictions.append(pred)
        return predictions

# Train the model
model = MovieRecommendationSystem(n_components=10)
model.fit(train_matrix)

# Make predictions on test set
test_predictions = model.predict_all(test_data)

# Calculate evaluation metrics
mse = mean_squared_error(test_data['rating'], test_predictions)
mae = mean_absolute_error(test_data['rating'], test_predictions)
rmse = np.sqrt(mse)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Visualize results
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted ratings
plt.subplot(1, 2, 1)
plt.scatter(test_data['rating'], test_predictions, alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--', lw=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings')
plt.grid(True, alpha=0.3)

# Plot 2: Rating distribution
plt.subplot(1, 2, 2)
plt.hist(test_data['rating'], bins=5, alpha=0.7, label='Actual', range=(0.5, 5.5))
plt.hist(test_predictions, bins=5, alpha=0.7, label='Predicted', range=(0.5, 5.5))
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Function to get movie recommendations for a user
def get_recommendations(user_id, model, user_movie_matrix, top_n=5):
    user_idx = user_id - 1
    if user_idx >= len(model.user_factors):
        print(f"User {user_id} not found")
        return []
    
    # Get movies the user hasn't rated
    user_ratings = user_movie_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Predict ratings for unrated movies
    recommendations = []
    for movie_id in unrated_movies:
        movie_idx = movie_id - 1
        predicted_rating = model.predict(user_idx, movie_idx)
        recommendations.append((movie_id, predicted_rating))
    
    # Sort by predicted rating and return top N
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Example: Get recommendations for user 1
user_id = 1
recommendations = get_recommendations(user_id, model, user_movie_matrix, top_n=5)
print(f"\nTop 5 movie recommendations for User {user_id}:")
for movie_id, predicted_rating in recommendations:
    print(f"Movie {movie_id}: Predicted rating {predicted_rating:.2f}")

# Display some statistics
print(f"\nDataset Statistics:")
print(f"Average rating: {ratings['rating'].mean():.2f}")
print(f"Rating distribution:")
print(ratings['rating'].value_counts().sort_index())
