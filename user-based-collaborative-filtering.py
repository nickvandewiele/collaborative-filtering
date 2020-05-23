
# coding: utf-8

# data manipulation
import pandas as pd
import numpy as np

from utils import read_ratings
from utils import read_names
from utils import pearson_similarity

# paths to the files. download them from http://files.grouplens.org/datasets/movielens/ml-100k.zip if you don't have them yet
MOVIE_RATINGS_PATH = 'ml-100k/u.data'
MOVIE_NAMES_PATH = 'ml-100k/u.item'

# number of neighbors of user
N_NEIGHBORS = 10

# number of recommendations
N_RECOMMENDATIONS = 5


def compute_similarities(user_id, ratings_matrix):
    """
    Compute the similarity of a given user with all the other users in the dataset.
    
    Remove the similarity value for the given user from the result.
    
    returns:
        - a pd.Series with the user id's as index, and similarity as series values
    """
    
    # get ratings of user to re-use in the similarity computation
    ratings_user = ratings_matrix.loc[user_id,:]
    
    # calculate the similarity between the given user and the other users
    similarities = ratings_matrix.apply(
        lambda row: pearson_similarity(ratings_user, row), 
        axis=1)

    similarities = similarities.to_frame(name='similarity')

    # find most similar users to the given user
    similarities = similarities.sort_values(by='similarity', ascending=False)
    
    # drop the similarity of the user (should be ~1 anyways)
    similarities = similarities.drop(user_id)
    
    return similarities


def predict_rating(item_id, ratings, similarities, N=10):
    """
    Predict the rating of a given item by a user, given the ratings of similar users.
    Takes the N users with the highest similarity measure, AND who have rated the given item.
    Returns the average rating of the most similar users who previously rated the item.
    
    parameters:
    - item_id: int, item that needs a rating prediction
    - ratings: pd.DataFrame
    - similarities: pd.DataFrame
    - N: int, number of neighbors to use for rating prediction
    
    returns:
    - a float representing the predicted rating for the given item
    
    """
    
    # get the ratings of all users for the specific item
    users_ratings = ratings.loc[:, item_id]
    
    # only keep users who rated the given item, otherwise you won't be able to generate a prediction based on the users ratings
    most_similar_users_who_rated_item = similarities.loc[~users_ratings.isnull()]
    
    # keep N users with highest similarities to given user who also rated the given item
    N_most_similar_users = most_similar_users_who_rated_item.head(N)
    
    # find ratings item for most similar users:
    ratings_for_item = ratings.loc[N_most_similar_users.index, item_id]
    
    # predict the rating of the item by averaging the ratings of that item of the most similar users
    return ratings_for_item.mean()


def recommend(user_id, ratings, movie_names, n_neighbors=10, n_recomm=5):
    """
    
    Recommend N movies for a given user based on ratings data.
    
    1. get the ratings of the user
    2. get the movies that the user has not rated
    3. compute the similarities between the user and the other users
    4. generate movie ratings predictions for the user based on the similarities with other users
    5. find the N movies with the highest predicted ratings
    
    parameters:
    - user_id: int, user to generate recommendations for
    - ratings: pd.DataFrame, user-movie ratings
    - movie_names: dict, mapping of (movie id -> movie name)
    - n_neighbors: int: the number of neighbors to use to generate rating predictions
    - n_recomm: int, number of movies to recommend
    
    returns:
    - pd.DataFrame with [movie_id, rating, movie name]
    
    """
    
    # all the items a user has not rated, that can be recommended
    all_items = ratings.loc[user_id,:]
    unrated_items = all_items.loc[all_items.isnull()]
    
    # convert the index with item ids into Series values
    unrated_items = unrated_items.index.to_series(name='item_ids').reset_index(drop=True)
    print('User {} has {} unrated items.'.format(user_id, len(unrated_items)))
    
    # compute user similarities
    similarities = compute_similarities(user_id, ratings)
        
    # generate predictions for unseen items based on the user similarity data
    predictions = unrated_items.apply(lambda d: predict_rating(d, ratings, similarities, N=n_neighbors))
    
    # sort items by highest predicted rating
    predictions = predictions.sort_values(ascending=False)
    
    # recommend top N items
    recommends = predictions.head(n_recomm)
    
    # reformat the result
    recommends = recommends.to_frame(name='predicted_rating')
    recommends = recommends.rename_axis('movie_id')
    recommends = recommends.reset_index()
    
    recommends['name'] = recommends.movie_id.apply(lambda d: movie_names[d])
    
    return recommends


def predict(user_id, item_id, ratings):
    """
    Make a prediction of the rating of an item for a user based on ratings data.

    Parameters:
    - user_id: int, id of the user in the dataset
    - item_id: int, id of the item in the dataset
    - ratings: pd.DataFrame, ratings dataset

    Returns:
    - prediction: float

    """

    # compute user similarities
    similarities = compute_similarities(user_id, ratings)
    
    prediction = predict_rating(item_id, ratings, similarities, N=N_NEIGHBORS)
    
    return prediction


if __name__ == '__main__':

    # read ratings data
    ratings = read_ratings(MOVIE_RATINGS_PATH)
    ratings = pd.DataFrame(data=ratings, columns=['user', 'movie', 'rating'])
    ratings = ratings.astype(int)

    # take a sample user, item
    sample = ratings.sample(random_state=42)
    user_id = sample.user.values[0]
    item_id = sample.movie.values[0]

    # convert long to wide
    ratings = ratings.pivot(index='user', columns='movie', values='rating')

    # read movie names
    movie_names = read_names(MOVIE_NAMES_PATH)

    # make a prediction for a specific user of a specific movie
    prediction = predict(user_id, item_id, ratings)
    print('Prediction for user {} of {}: {}'.format(user_id, movie_names[item_id], prediction))

    # recommend
    recommends = recommend(user_id, ratings, movie_names, n_neighbors=N_NEIGHBORS, n_recomm=N_RECOMMENDATIONS)
    print(recommends)

    print('Done recommending!')

    