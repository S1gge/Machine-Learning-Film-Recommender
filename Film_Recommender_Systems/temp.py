import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache

@cache
def load_data_files():
    movies = pd.read_csv('./Film_Recommender_Systems/CSV/movies.csv')
    ratings = pd.read_csv('./Film_Recommender_Systems/CSV/ratings.csv')
    #tags = pd.read_csv('./CSV/tags.csv')
    new_movies = movies.drop_duplicates("title").copy()
    title_sep = new_movies["title"].str.rsplit(" ", n=1, expand = True)
    new_movies["movie title"] = title_sep[0]
    new_movies["year"] = title_sep[1]
    new_movies.drop(columns=["title"], inplace=True)

    movies_with_ratings = ratings.merge(new_movies, on="movieId")
    movies_with_ratings.drop(["timestamp"], axis=1, inplace=True)
    movies_with_ratings.fillna('(Release year unknown)', inplace=True)
    return new_movies, movies_with_ratings

def extract_features(movies_with_ratings):
    x = movies_with_ratings.groupby("userId").count()["rating"] > 100
    expert_users = x[x].index
    filtered_user_ratings = movies_with_ratings[movies_with_ratings["userId"].isin(expert_users)]
    y = filtered_user_ratings.groupby("movie title").count()["rating"] >= 50
    famous_movies = y[y].index
    user_ratings = filtered_user_ratings[filtered_user_ratings["movie title"].isin(famous_movies)]
    design_matrix = user_ratings.pivot_table(index="movie title", columns="userId", values="rating")
    design_matrix.fillna(0, inplace=True)
    return design_matrix

def make_model(design_matrix):
    scaler = StandardScaler(with_mean=True, with_std=True)
    design_matrix_centered = scaler.fit_transform(design_matrix)
    similiarity_score = cosine_similarity(design_matrix_centered)
    return similiarity_score

def recommend(movie_name, new_movies, design_matrix, similiarity_score):
    index = np.where(design_matrix.index==movie_name)[0][0]
    similar_movies = sorted(list(enumerate(similiarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []

    for index, similarity in similar_movies:
        item = []
        temp_df = new_movies[new_movies["movie title"]==design_matrix.index[index]]
        item.extend(temp_df["movie title"].values)
        item.extend(temp_df["year"].values)
        data.append(item)
    return data


def main(name):
    new_movies, movies_with_ratings = load_data_files()
    design_matrix = extract_features(movies_with_ratings)
    model = make_model(design_matrix)
    print(recommend(name, new_movies, design_matrix, model))

if __name__ == '__main__':
    name = input("Input a movie name: ")
    main(name)