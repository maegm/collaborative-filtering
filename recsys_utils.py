import numpy as np
import pandas as pd


def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalize_ratings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R, axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return Ynorm, Ymean


def load_precalc_params_small():

    url_x = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movies_X.csv'
    X = (
        pd.read_csv(url_x, header=None)
        .to_numpy()
    )

    url_w = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movies_W.csv'
    W = (
        pd.read_csv(url_w, header=None)
        .to_numpy()
    )

    url_b = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movies_b.csv'
    b = (
        pd.read_csv(url_b, header=None)
        .to_numpy()
        .reshape(1, -1)
    )

    num_movies, num_features = X.shape
    num_users, _ = W.shape
    return X, W, b, num_movies, num_features, num_users

    
def load_ratings_small():
    url_y = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movies_Y.csv'
    Y = (
        pd.read_csv(url_y, header=None)
        .to_numpy()
    )

    url_r = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movies_R.csv'
    R = (
        pd.read_csv(url_r, header=None)
        .to_numpy()
    )
    return Y, R


def load_movie_list_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    url = 'https://raw.githubusercontent.com/maegm/collaborative-filtering/master/data/small_movie_list.csv'
    df = pd.read_csv(url, header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return mlist, df
