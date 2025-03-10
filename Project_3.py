import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('Project_3_movies.csv')
ratings = pd.read_csv('Project_3_ratings.csv')

data = pd.merge(movies, ratings, on='movieId')
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
similarity_scores = cosine_similarity(user_item_matrix.fillna(0))

similar_movies = np.argsort(-similarity_scores[0])
top_10_similar_movies = user_item_matrix.columns[similar_movies[:10]]



user_item_matrix_T = user_item_matrix.T
user_item_matrix_T = user_item_matrix_T.fillna(0)
similarity_scores = cosine_similarity(user_item_matrix_T)

def recommend_movies(movie_title):
    try:
        movie_index = list(user_item_matrix_T.index).index(movie_title)
    except ValueError:
        return "Movie not found in data."
    similar_movies = np.argsort(-similarity_scores[movie_index])[::-1]
    top_10_similar_movies = user_item_matrix_T.index[similar_movies[:3]]
    return ', '.join(top_10_similar_movies.tolist())

print('1. ' + recommend_movies("Toy Story (1995)"))
print('2. ' + recommend_movies("Jumanji (1995)"))

def recommend_movies_to_new_user():
    most_rated_movies = user_item_matrix.count().sort_values(ascending=False)
    return ', '.join(most_rated_movies.index[:3].tolist())

def recommend_new_movies(user_id):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    highly_rated_movies = user_ratings[user_ratings >= 4].index
    new_movies = movies[~movies['title'].isin(user_item_matrix.columns)]
    similar_movies = []
    for movie in highly_rated_movies:
        movie_genres = movies[movies['title'] == movie]['genres'].iloc[0]
        similar_movies.extend(new_movies[new_movies['genres'] == movie_genres]['title'].tolist())
    similar_movies = list(set(similar_movies))
    return ', '.join(similar_movies[:3])

print('Recommendations for new user:')
print(recommend_movies_to_new_user())

print('Recommendations for user 1 with new movies:')
print(recommend_new_movies(1))