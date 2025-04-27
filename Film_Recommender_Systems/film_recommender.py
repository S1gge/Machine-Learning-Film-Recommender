import pandas as pd
import numpy as np
import os
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache
def load_data_files():
    movies = pd.read_csv('./Film_Recommender_Systems/CSV/movies.csv')
    tags = pd.read_csv('./Film_Recommender_Systems/CSV/tags.csv')
    #ratings = pd.read_csv('./CSV/ratings.csv')
    return movies, tags


def extract_features(movies, tags):
    new_movies = movies.drop_duplicates("title").copy()
    title_sep = new_movies["title"].str.rsplit(" ", n=1, expand = True)
    new_movies["movie title"] = title_sep[0]
    new_movies["year"] = title_sep[1]
    new_movies.drop(columns=["title"], inplace=True)
    new_movies.fillna('(Release year unknown.)', inplace=True)
    new_movies["genres"] = new_movies["genres"].str.replace('|', ' ', regex=False).str.lower()

    tags.drop(columns=["userId", "timestamp"], inplace=True, errors = 'ignore')
    tags.dropna(inplace=True)
    tags.drop_duplicates(inplace=True)
    tags["tag"] = tags["tag"].str.lower()

    movies_with_tags = new_movies.merge(tags, on="movieId")

    tag_counts = movies_with_tags.groupby("movieId")["tag"].count()
    sorted_movieId = tag_counts[tag_counts>=30].index
    movies_with_tags = movies_with_tags[movies_with_tags["movieId"].isin(sorted_movieId)]

    tags_clean = movies_with_tags.groupby("movieId")[["tag"]].agg(lambda x:' '.join(x))
    movies_filtered = new_movies.merge(tags_clean, on="movieId")

    movies_filtered['description'] = movies_filtered[['genres', 'tag']].astype(str).agg(' '.join, axis=1)
    movies_filtered = movies_filtered.drop(["genres", "tag"], axis=1).copy()

    return movies_filtered


def make_model(movies_filtered):
    count_vectorizer = CountVectorizer(stop_words='english', max_df=0.8, min_df=2)
    bag_of_words = count_vectorizer.fit_transform([text for text in movies_filtered['description']])

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bag_of_words).toarray()

    similarity_score = cosine_similarity(tfidf_matrix)
    return similarity_score


def recommend(movie_name, movies_filtered, similarity_score):
    movie_name_lower = movie_name.lower()
    movies_filtered_lower = movies_filtered['movie title'].str.lower()
    
    if movie_name_lower not in movies_filtered_lower.values:
        return "Movie not found"
    
    index = movies_filtered[movies_filtered_lower == movie_name_lower].index[0]

    similar_movies = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i, _ in similar_movies:
        item = []
        temp_df = movies_filtered.iloc[i]
        item.append(temp_df["movie title"])
        item.append(temp_df["year"])
        data.append(item)
        df_output = pd.DataFrame(data, columns=['Movie Title', 'Release Year'])
    return df_output

def main(movie_name):
    movies, tags = load_data_files()
    movies_filtered = extract_features(movies, tags)
    similarity_score = make_model(movies_filtered)
    print(recommend(movie_name, movies_filtered, similarity_score))

if __name__ == '__main__':
    print("\n***** Welcome To Film Recommender *****")
    
    while True:
        movie_name = input("\nEnter a movie name: ")
        print("\nYou might be intrested in:\n")
        main(movie_name)
        choice = input("\nDo you want to try again? (Y/N) ")
        if choice == "N" or choice == "n":
            break
        os.system('cls')
    os.system('cls')
    print("\nThank you for this time!\n")