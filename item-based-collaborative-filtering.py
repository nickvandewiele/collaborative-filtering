
# coding: utf-8

# data manipulation
import pandas as pd

# imports to construct the similarity function
from numpy import dot
from numpy.linalg import norm

from utils import read_ratings
from utils import read_names


# paths to the files. download them from http://files.grouplens.org/datasets/movielens/ml-100k.zip if you don't have them yet
MOVIE_RATINGS_PATH = 'ml-100k/u.data'
MOVIE_NAMES_PATH = 'ml-100k/u.item'
RECOMMENDER_PATH = 'result.csv'


def similarity(group_of_movie_ratings):
    """
    
    Cosine similarity measure, implemented through numpy functions.
    
    Accepts a DataFrame with 2 columns: ratings_1, ratings_2
    
    Returns a float
    """
    a = group_of_movie_ratings['rating_1'].values
    b = group_of_movie_ratings['rating_2'].values

    cos_sim = dot(a, b)/(norm(a)*norm(b))

    return cos_sim


def load_recommender(recommender_path):

    df = pd.read_csv(recommender_path)
    return df


def recommend(
    movie_id, 
    ratings, 
    names, 
    min_cooccurrence=50, 
    top=5):
    """
    
    Make movie recommendations for the given movie.
    
    Input:
    - ratings: the movie ratings with similarity measures
    - movie_id: the movie id for which you want recommendations
    - names: a mapping of movie id -> movie name
    - min_count: the minimum threshold for co-occurrence
    - top: the number of recommendations to make
    
    Returns:
    None
    """
    
    # filter based on given movie, minimum occurrence
    recomm = ratings.loc[
        ((ratings.movie_1 == movie_id) | \
        (ratings.movie_2 == movie_id)) & \
        (ratings.cooccurrence >= min_cooccurrence)]
    
    # sort by similarity measure
    recomm = recomm.sort_values(by=['similarity'], ascending=False)
    
    # keep top n recommendations
    recomm = recomm.head(top).reset_index(drop=True)
    
    # edge cases
    if len(recomm) == 0:
        print('No recommendations for movie id {}...'.format(movie_id))
        return
    else:
        # check if the recommended movie id corresponds to first or second movie
        recomm['recommended_movie'] = recomm.apply(lambda row: row.movie_2 if row.movie_1 == movie_id 
                                          else row.movie_1 , axis=1)
        
        recomm['recommended_movie'] = recomm['recommended_movie'].astype(int)
        
        # find movie title for the movie_id
        recomm['title'] = recomm['recommended_movie'].apply(lambda d: names[d])
                        
        return recomm[['recommended_movie', 'title', 'similarity', 'cooccurrence']]


def train():

    ratings = read_ratings(MOVIE_RATINGS_PATH)
    ratings = pd.DataFrame(data=ratings, columns=['user', 'movie', 'rating'])
    ratings = ratings.astype(int)

    ratings_joined = ratings.merge(ratings, on='user', suffixes=('_1', '_2'))

    ratings_filtered = ratings_joined[ratings_joined['movie_1'] < ratings_joined['movie_2']]
    ratings_filtered = ratings_filtered.drop(['user'], axis=1)

    ratings_grouped = ratings_filtered.groupby(by=['movie_1', 'movie_2'])

    cooccurrence = ratings_grouped.size().to_frame('cooccurrence')

    ratings_similar = ratings_grouped.apply(similarity)
    ratings_similar = ratings_similar.to_frame('similarity')

    df = pd.concat([ratings_similar, cooccurrence], axis=1)

    df = df.reset_index().sort_values(by=['cooccurrence', 'similarity'], ascending=False)
    df.to_csv(RECOMMENDER_PATH, index=False)

    print('Done training!')



def predict():

    ratings = load_recommender(RECOMMENDER_PATH)
    movie_names = read_names(MOVIE_NAMES_PATH)
    movie_id = 10
    min_cooccurrence=50
    top=5
    recommendations = recommend(movie_id, ratings, movie_names, min_cooccurrence, top)

    print('Recommendations for {}'.format(movie_names[movie_id]))
    print(recommendations)

    print('Done recommending!')


if __name__ == '__main__':
    train()
    predict()