#import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = ({'movie_id': [1, 2, 3, 4, 5],
        'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
        'genre': ['Action,Sci-Fi', 'Comedy,Thriller', 'Action,Crime', 'Comedy,Drama', 'Action,Sci-Fi'],
        'director': ['Director 1', 'Director 2', 'Director 3', 'Director 4', 'Director 5'],
        'cast': ['Actor 1, Actor 2', 'Actor 3, Actor 4', 'Actor 5, Actor 6', 'Actor 7, Actor 8', 'Actor 9, Actor 10']})

#convert the data into a pandas dataframe
df = pd.DataFrame(data)

#display the dataset
#print(df)

#define a TF-IDF vectorizer to transform genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

#Fit and transform the genre text into a matrix of TF-IDF features. Similarity is  based on genre.
tfidf_matrix = tfidf.fit_transform(df['genre'])

#Compute the cosine similarity between the TF-IDF features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Function to get movie recommendations based on genre/cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    #Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]
    
    #Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Get the scores of the 2 most similar movies
    sim_scores = sim_scores[1:3]
    
    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    #Return the top 2 most similar movies
    return df['title'].iloc[movie_indices]
    
#Test the function
recommended_movies = get_recommendations('Movie 1')
print(f"Recommended movies for 'Movie 1': {recommended_movies}")
for movie in recommended_movies:
    print(movie)

