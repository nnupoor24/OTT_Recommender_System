import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie dataset
movies = pd.read_csv("data/movies.csv")


# Replace missing values
movies["genres"] = movies["genres"].fillna("")


# Convert genres into vectors
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["genres"])


# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Recommendation function
def recommend_movies(movie_title):

    # Check if movie exists
    if movie_title not in movies["title"].values:
        return "Movie not found in dataset"

    # Find movie index
    idx = movies[movies["title"] == movie_title].index[0]

    # Similarity scores
    similarity_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Top 5 movies
    similarity_scores = similarity_scores[1:6]

    # Movie indexes
    movie_indices = [i[0] for i in similarity_scores]

    # Return movie titles
    return movies["title"].iloc[movie_indices]


# USER INPUT
movie_name = input("Enter movie name: ")


print("\nRecommended Movies:\n")

print(recommend_movies(movie_name))