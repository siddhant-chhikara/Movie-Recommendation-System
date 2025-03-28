•	Like YouTube or other movie streaming platforms, this project aims to create a personalized movie recommendation system to enhance user experience by suggesting movies based on their preferences

•	Loaded two datasets containing information related to movies and ratings. Merged both datasets to create a unified dataset and then used this data to create user-item matrix. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is used to measure how similar two documents (or items, in the context of a recommendation system) are irrespective of their size.

•	In the context of a movie recommendation system, cosine similarity is used to calculate the similarity between different movies based on user ratings. The idea is that if two movies are rated similarly by the same users, then those movies are similar. Calculated similarity using cosine similarity between each pair of movies, while filling missing values with 0. Sorted movies based on their similarity scored and selected top 10 movies as recommendations. 

•	The cold start problem in recommendation systems refers to the difficulty of making accurate recommendations for users or items that have little to no interaction data. In the provided code, the cold start problem is addressed in two ways:

I.	For New Users: The `recommend_movies_to_new_user()` function recommends the three most rated movies to new users. This is a common approach to the cold start problem for users, as popular items are often a safe recommendation.

II.	For New Movies: The `recommend_new_movies(user_id) ` function recommends new movies to a user based on the genres of movies they have highly rated in the past. This is a form of content-based filtering, which can be used to recommend items without interaction data by using item features. In this case, the genre of the movie is used as the feature.

