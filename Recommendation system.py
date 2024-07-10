#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample data: movies with their genres
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genres': ['Action|Adventure', 'Adventure|Fantasy', 'Action|Thriller', 'Fantasy|Drama', 'Drama|Romance']
}

# Convert data to DataFrame
movies_df = pd.DataFrame(data)
print("Movies DataFrame:")
print(movies_df)

# Create a CountVectorizer to convert genres to a matrix of token counts
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genres_matrix = vectorizer.fit_transform(movies_df['genres'])
print("\nGenres Matrix:")
print(genres_matrix.toarray())

# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(genres_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies_df['title'], columns=movies_df['title'])
print("\nMovie Similarity Matrix:")
print(movie_similarity_df)

# Function to get recommendations for a specific movie
def get_movie_recommendations(movie_title, movie_similarity_df, movies_df, num_recommendations=2):
    # Get the similarity scores for the specified movie
    similarity_scores = movie_similarity_df[movie_title]

    # Sort the scores in descending order and get the top N recommendations
    top_recommendations = similarity_scores.sort_values(ascending=False)[1:num_recommendations+1]
    return top_recommendations

# Get recommendations for 'Movie A'
movie_title = 'Movie A'
recommendations = get_movie_recommendations(movie_title, movie_similarity_df, movies_df, num_recommendations=2)
print(f"\nRecommendations for '{movie_title}':")
print(recommendations)


# In[ ]:




