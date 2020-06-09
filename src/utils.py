def read_ratings(path_to_ratings):
    """
    Read the raw data of the movie ratings.
    
    Returns a list of tuples:
    (user id, movie id, rating)
    """

    data = []
    with open(path_to_ratings) as f:
        for line in f:
            # user id | item id | rating | timestamp
            pieces = line.split()
            user_id = int(pieces[0])
            movie_id = int(pieces[1])
            rating = float(pieces[2])
            data.append((user_id, movie_id, rating))
        
    return data


def read_names(path_to_names):
    """
    Read the mapping of movie id -> movie name
    
    Returns a dictionary
    {movie id -> movie name}
    """

    data = {}
    with open(path_to_names) as f:
        for line in f:
            # movie id | movie title | ...
            pieces = line.split('|')
            movie_id = int(pieces[0])
            title = pieces[1]
            data[movie_id] = title
        
    return data


def pearson_similarity(v1, v2):
    """
    Compute the Pearson correlation between to ratings vectors.
    
    pd.corr() function can handle missing data.
    
    parameters: 
    - v1, v2: pd.Series, ratings vectors
    
    returns:
    - float
    
    """
    
    pearson = v1.corr(v2)
    
    return pearson