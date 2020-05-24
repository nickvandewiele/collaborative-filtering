
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
SIMILARITIES_PATH = 'result.csv'

# minimum co-occurrence of movie pairs
MIN_OCCURRENCES = 50

# number of recommendations
N_RECOMMENDATIONS = 5


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


def load_item_similarities(similarities_path):
    """
    Reads a csv  with movie pair similarities and co-occurrences.

    Returns:
    - pd.DataFrame with columns [movie a, movie b, similarity, co-occurrence]

    """

    df = pd.read_csv(similarities_path)
    return df


def recommend(
    movie_id, 
    similarities, 
    names, 
    min_cooccurrence=50, 
    top=5):
    """
    
    Make movie recommendations for the given movie.
    
    Input:
    - similarities: the movie ratings with similarity measures
    - movie_id: the movie id for which you want recommendations
    - names: a mapping of movie id -> movie name
    - min_count: the minimum threshold for co-occurrence
    - top: the number of recommendations to make
    
    Returns:
    - pd.DataFrame with columns [recommended movie, title, similarity, co-occurrence]

    """
    
    # filter based on given movie, minimum occurrence
    recomm = similarities.loc[
        ((similarities.movie_1 == movie_id) | \
        (similarities.movie_2 == movie_id)) & \
        (similarities.cooccurrence >= min_cooccurrence)]
    
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


def compute_similarities(ratings):
    """
    Calculate item-item similarities based on ratings data.

    Parameters:
    - ratings: pd.DataFrame, ratings in the long form (user_id, item_id, rating)

    Returns:
    - similarities: pd.DataFrame, in the form [item id 1, item id 2, similarity, co-occurrence]

    """

    # inner-join to create movie pairs
    ratings_joined = ratings.merge(ratings, on='user', suffixes=('_1', '_2'))

    # filter useless movie pairs
    ratings_filtered = ratings_joined[ratings_joined['movie_1'] < ratings_joined['movie_2']]
    ratings_filtered = ratings_filtered.drop(['user'], axis=1)

    # group ratings for the same movie pair
    ratings_grouped = ratings_filtered.groupby(by=['movie_1', 'movie_2'])

    # count co-occurrence for each movie pair
    cooccurrence = ratings_grouped.size().to_frame('cooccurrence')

    # calculate similarity for each movie pair
    ratings_similar = ratings_grouped.apply(similarity)
    ratings_similar = ratings_similar.to_frame('similarity')

    # put everything together, reformat
    df = pd.concat([ratings_similar, cooccurrence], axis=1)

    df = df.reset_index().sort_values(by=['cooccurrence', 'similarity'], ascending=False)

    return df



def predict(user_id, item_id, ratings, similarities, min_cooccurrence=5, top=5):
    """
    Make a prediction of the rating of an item for a user based on ratings data.

    If the item was already rated by the user, simply return the rating.
    If not:
    - find the most similar items, but only keep items above a minimum co-occurrence threshold.
    - only keep item pairs of items that have been rated by the user already
    - sort the items based on most similar, and keep the top n of item pairs
    - give a rating based on the average rating that the user gave to the list of most similar items.

    Parameters:
    - user_id: int, id of the user in the dataset
    - item_id: int, id of the item in the dataset
    - ratings: pd.DataFrame, ratings in the long form (user_id, item_id, rating)
    - similarities: pd.DataFrame, in the form [item id 1, item id 2, similarity, co-occurrence]

    Returns:
    - prediction: float, predicted rating by the user for the item

    """

    # search for (user, item) in existing ratings
    existing_ratings = ratings.loc[(ratings.user == user_id) & (ratings.movie == item_id)]

    if len(existing_ratings) > 0:
        print('Found existing ratings:\n{}'.format(existing_ratings))
        # take first match
        return existing_ratings.rating.values[0]

    # find most similar movie
    
    # filter based on given movie, minimum occurrence
    recomm = similarities.loc[
        ((similarities.movie_1 == item_id) | \
        (similarities.movie_2 == item_id)) & \
        (similarities.cooccurrence >= min_cooccurrence)]

    # remove movie pairs that have not been rated by the user
    items_user = ratings.loc[ratings.user == user_id]
    print('User {} has rated {} items.'.format(user_id, len(items_user)))

    recomm = recomm.loc[recomm.movie_1.isin(items_user.movie) | recomm.movie_2.isin(items_user.movie)]
    print('After removing movies that have not been rated by the user, we have {} items.'.format(len(recomm)))

    # if there are no other rated items, we cannot make a prediction
    if len(recomm) == 0:
        return -1

    # sort by similarity measure
    recomm = recomm.sort_values(by=['similarity'], ascending=False)
    
    # keep top n recommendations
    recomm = recomm.head(top).reset_index(drop=True)
    
    # get the list of similar movies
    recomm['movie'] = recomm.apply(lambda row: row.movie_2 if row.movie_1 == item_id 
                                          else row.movie_1 , axis=1)
    recomm['movie'] = recomm['movie'].astype(int)

    recomm = recomm[['movie', 'similarity']]

    # get the rating that the user gave to the similar movies
    recomm['rating'] = recomm.movie.apply(
        lambda idx: ratings.loc[(ratings.user == user_id) & (ratings.movie == idx)].rating.values[0])
    
    # print(recomm)

    # use the average rating of similar movies seen by the user as a prediction
    predicted_rating = recomm.rating.mean()

    return predicted_rating


if __name__ == '__main__':
    
    ratings = read_ratings(MOVIE_RATINGS_PATH)
    ratings = pd.DataFrame(data=ratings, columns=['user', 'movie', 'rating'])
    ratings = ratings.astype(int)

    print(ratings.head())

    # compute_similarities
    similarities = compute_similarities(ratings)
    similarities.to_csv(SIMILARITIES_PATH, index=False)
    print('Done computing similarities!')
    
    similarities = load_item_similarities(SIMILARITIES_PATH)
    movie_names = read_names(MOVIE_NAMES_PATH)
    
    print(similarities.head())
    
    user_id = 381
    movie_id = 12
    

    # existing (user, item) pair:
    user_id = 196
    movie_id = 242
    
    # recommend
    recommendations = recommend(movie_id, similarities, movie_names, min_cooccurrence=MIN_OCCURRENCES, top=N_RECOMMENDATIONS)

    print('Recommendations for {}'.format(movie_names[movie_id]))
    print(recommendations)
    print('Done recommending!')
    
    # predict
    prediction = predict(user_id, movie_id, ratings, similarities)
    print('Prediction for user {} of {}: {}'.format(user_id, movie_names[movie_id], prediction))
