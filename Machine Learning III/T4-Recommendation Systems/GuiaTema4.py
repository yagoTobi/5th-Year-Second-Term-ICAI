# * ################################################################################################
# * ##############     Handy Guide ICAI - MACHINE LEARNING III   - Yago Tobio Souto  ###############
# * ################################################################################################

# * Librerias
import os
import sys
import math
import cornac
import operator
import itertools
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
import matplotlib.pyplot as plt

from cornac.utils import cache
from adjustText import adjust_text
from scipy.sparse.linalg import svds
from numpy.linalg import matrix_rank
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from sklearn.metrics import mean_squared_error
from cornac.models import MF, NMF, BaselineOnly
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
VERBOSE = False

# * Import file
df_path = "df_path"
df = pd.read_csv(df_path, header=0)

# * EDA
df.head()
df.info()
df.shape()
column_unique_values = df["column"].unique()
number_column_unique_values = df["column"].nunique()

# ? - Agrupación de columnas:
group_by_and_count = pd.DataFrame(df.groupby("ProductId")["Rating"].count())
sorted_values_by_criteria = group_by_and_count.sort_values("Rating", ascending=False)
sorted_values_by_criteria.head(10)

# * Take a sample of the dataset
df_sample = df.sample(n=10000, random_state=42)


# ! Recommender systems:
# ? - Collaborative Filtering <- WE ARE INTERESTED IN THIS ONE
# * - Model-based filtering technique <- Clustering, NN, Association, etc...
# ? Use ML to find user ratings of unrated items: PCA, SVD, Neural nets
# ? Performance reduction with sparse data.
# * - Memory-based filtering technique  <- User Based / Item Based <- THIS ONE FOR THE PRACTICAL
# ? Based on cosine similarity or pearson correlation and taking the avg. of ratings.
# ? Non-scalable for sparse data.

# * #######################################################################################
# * #######################################################################################
# * #######################################################################################
# * ### Indice (Collaborative Filtering techniques)                                    ####
# * ### 1. Memory-Based Filtering                  (Linea XX)                          ####
# * ###    a. User-Based Filtering                 (Linea XX)                          ####
# * ###    b. Item-Based Filtering                 (Linea XXX)                         ####
# * ### 2. Model-Based Collaborative Filtering                                         ####
# * ###    a. Matrix Factorisation                                                     ####
# * ###    b. Singular Value Decomposition                                             ####
# * #######################################################################################
# * #######################################################################################
# * #######################################################################################

# ? - Content-Based Filtering
"""
! IMPORTANTE, ESTAMOS ASUMIENDO QUE EN CADA TABLA HAY: userId, itemId, rating, timestamp
! Asegurate que tienes una chuleta de pandas para eliminar las columnas 
! Tambien nos podríamos enfrentar con tablas que tengan mas datos que puedan ser utiles. 
! Pero por ahora el foco de la asignatura esta con el user-item matrix. No por ejemplo movie genre.
"""

# ! - Memory-Based: User-Based Filtering
# * https://medium.com/@corymaklin/memory-based-collaborative-filtering-user-based-42b2679c6fb5
# As an example, we're going to take the movies dataset
# TODO: We're not going to get a test dataset in the exam, so what do we do there?
train_df = pd.read_csv(
    "ml-100k/u1.base",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)
test_df = pd.read_csv(
    "ml-100k/u1.test",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)

train_df.head()

# * !!!! - Construct the ratings matrix:
# ? - values -> Cell values | index -> Rows | columns -> Columns
ratings_matrix = pd.pivot_table(
    train_df, values="rating", index="user_id", columns="item_id"
)

# * Normalise the ratings matrix by subtracting every user's rating by the mean users rating:
normalized_ratings_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis=1), axis=0)

# ? - Case 1. Pearson Correlation to determine similarity
similarity_matrix = (
    ratings_matrix.T.corr()
)  # * This determines the similarity of each user

# ? - Case 2. Cosine similarity (WE WOULD NOW HAVE TO IMPUTE THE MISSING DATA -> Most common method: Fill in with the user or item average rating)
# ?           We can proceed with this as long as all of the items have been normalised first.
# ? - If we want to fill it in with zeroes:
item_similarity_cosine = cosine_similarity(normalized_ratings_matrix.fillna(0))
item_similarity_cosine = cosine_similarity(
    normalized_ratings_matrix.fillna(ratings_matrix.T.mean()[user_id])  # type:ignore
)
item_similarity_cosine = cosine_similarity(
    normalized_ratings_matrix.fillna(ratings_matrix.T.mean()[item_id])  # type:ignore
)
item_similarity_cosine


# * Calculate the score according to the formula:
# ? - Case 1: Pearson
def calculate_score(user_id, item_id):
    """
    Understanding the score formula is very important:
    S(u,i) = r_u + {Sum(r_vi - r_v) * w_uv}/{Sum(w_uv)}
    Score for user_id, item i = avg user rating + (normalised v rating*similarity weight)/similarity weight
    Where:
     * r_u is the user u's average rating
     * r_vi is the user v's rating on item i
     * r_v is the user v's average rating
     * w_uv is the similarity between user's v and u (Done by either Pearson or Cosine)

    """

    # ? Check if the item is in the training dataset:
    if item_id not in ratings_matrix.columns:  #
        return 2.5

    similarity_scores = similarity_matrix[user_id].drop(
        labels=user_id
    )  # ? Take out the user itself, so that it doesn't self-match
    normalized_ratings = normalized_ratings_matrix[item_id].drop(
        index=user_id
    )  # ? For all of the ratings, take out the user itself

    # ? If none of the other users have rated items in common with the user, return the baseline value:
    if similarity_scores.isna().all():
        return 2.5

    total_score = 0
    total_weight = 0
    # ? For each item
    for v in normalized_ratings.index:
        # ? It's possible for a user to rate the item but not in common with the user in question:
        if not pd.isna(similarity_scores[v]):
            total_score += normalized_ratings[v] * similarity_scores[v]
            total_weight += abs(similarity_scores[v])

    avg_user_rating = ratings_matrix.T.mean()[user_id]
    return (
        avg_user_rating + total_score / total_weight
    )  # ? Adding back the original weight


# * Score testing
test_ratings = np.array(test_df["rating"])
user_item_pairs = zip(test_df["user_id"], test_df["item_id"])
pred_ratings = np.array(
    [calculate_score(user_id, item_id) for (user_id, item_id) in user_item_pairs]
)
print(np.sqrt(mean_squared_error(test_ratings, pred_ratings)))


# * Baseline rating - Whic is just the mean of all ratings given
baseline_rating = train_df["rating"].mean()
baseline_ratings = np.array([baseline_rating for _ in range(test_df.shape[0])])
print(np.sqrt(mean_squared_error(test_ratings, baseline_ratings)))

# ! - Memory-Based: Item-based filtering
# * Predict user's rating for one movie process:
# * 1. Create a list of the movies which the user 1 has watched and rated.
# * 2. Rank the similarities between the movies that user 1 has rated and the movie to predict.
# * 3. Select top n movies with the highest similarity scores.
# ? 4. Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1.
# Pick a user ID
picked_userid = 1

# Pick a movie
picked_movie = "American Pie (1999)"

# ? - Normalised dataframe which filters all of the movies which a specific user has watched
picked_userid_watched = (
    pd.DataFrame(
        normalized_ratings_matrix[picked_userid]
        .dropna(
            axis=0, how="all"
        )  # ? Here, we are removing the movies which the user hasn't watched.
        .sort_values(ascending=False)  # ? Sort the values in descending order
    )
    .reset_index()  # ? - Reset the endings
    .rename(columns={1: "rating"})
)

picked_userid_watched.head()

# ?####################### SINGLE ITEM EXAMPLE ##############################################
# ? Getting the similarity score for a specific movie using Pearson
item_similarity_p = normalized_ratings_matrix.T.corr()
item_similarity_cosine = cosine_similarity(normalized_ratings_matrix.fillna(0))

picked_movie_similarity_score = (
    item_similarity_p[
        [picked_movie]
    ]  # * This is a specific Pearson case, for a specific film
    .reset_index()
    .rename(columns={"American Pie (1999)": "similarity_score"})
)

n = 5  # ? - Get the first 5 films which user 1 rated
# ? Rank the similarities between the movies which user 1 rated and American Pie.
picked_userid_watched_similarity = pd.merge(
    left=picked_userid_watched,
    right=picked_movie_similarity_score,
    on="title",
    how="inner",  # * This inner join with pandas is the way in which we're going to get the films in common
).sort_values("similarity_score", ascending=False)[:n]

# ? - In case of the item_based_example, we can see that the similarity score tell us, that they are completely uncorrelated.
# ? - After detecting the similarity score for each specific film with respect to the others, the higher similarity movies get more weight.
# ? - This weighted average is the predicted rating for American Pie by user 1.
predicted_item_rating = round(
    # * So what this essentially does, is compute the average of the ratings
    # * However the average is weighted by how similar the films are. Why is that?
    # * This is because more similar films will be closer, so this is a pseudo-approximation.
    np.average(
        picked_userid_watched_similarity["rating"],
        weights=picked_userid_watched_similarity["similarity_score"],
    ),
    6,  # * Number of decimals
)

# * Don't we have to add the average user recommentadion, since this is based on the normalised dataset?
print(
    f"The predicted rating for {picked_movie} by user {picked_userid} is {predicted_item_rating}"
)
# ?########################?########################?########################?########################?#######################


# * Now we create a function, with follows four steps:
# ? - Create a list of movies which the user hasn't watched before.
# ? - Loop through this list of movies, and create a predicted score for each of these
# ? - Sort them with the predicted scores first.
# ? - Select the top k movies as recommendations for the target user.
def item_based_rec(
    picked_userid=1, number_of_similar_items=5, number_of_recommendations=3
):
    picked_userid_unwatched = pd.DataFrame(
        normalized_ratings_matrix[picked_userid].isna()  # ? - Get the unwatched movies
    ).reset_index()

    # ? - From the dataset of the unwatched films just get the first column
    picked_userid_unwatched = picked_userid_unwatched[
        picked_userid_unwatched[1] == True
    ]["title"].values.tolist()

    # ? - Movies which the user has watched
    picked_userid_watched = (
        pd.DataFrame(
            normalized_ratings_matrix[picked_userid]
            .dropna(axis=0, how="all")
            .sort_values(ascending=False)
        )
        .reset_index()
        .rename(columns={1: "rating"})  # ? Rename the second column to get the rating.
    )

    # ? - Dictionary to save the unwatched movie and predicted rating pair
    rating_prediction = {}

    # * Calculate the similarity score of an indiv. movie with the list.
    for movie in picked_userid_unwatched:
        picked_movie_similarity_score = (
            item_similarity_p[[movie]]
            .reset_index()
            .rename(columns={picked_movie: "similarity_score"})
        )

        picked_userid_watched_similarity = pd.merge(
            left=picked_userid_watched,
            right=picked_movie_similarity_score,
            on="title",
            how="inner",
        ).sort_values("similarity_score", ascending=False)[:number_of_similar_items]

        # ? - Calculate the predicted rating using weighted average of similarity scores
        # ? - and the user 1 ratings
        predicted_rating = round(
            np.average(
                picked_userid_watched_similarity["rating"],
                weights=picked_userid_watched_similarity["similarity_score"],
            ),
            6,
        )
        # ? Save the predicted rating in the dictionary
        rating_prediction[picked_movie] = predicted_rating
    # ? - The key=operator.itemgetter(1), just tells the sorted function to get the rating_prediction second column in desc. order
    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[
        :number_of_recommendations
    ]


# ! - Model-based collaborative filtering techniques:

# * Proceed with the steps to obtain the ratings matrix
mean_rating = 2.5
r_df = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(
    mean_rating
)

# ? - Pass the dataframe into a numpy df
r = r_df.to_numpy()

# ? - Center the ratings by subtracting the overall mean of the matrix
user_ratings_mean = np.mean(r, axis=1)
r_centered = r - user_ratings_mean.reshape(-1, 1)
print(r.shape)  # * Returns dimension
print(
    np.count_nonzero(r)
)  # * This returns the nonzero elements -> Check that all elements are filled in.

#! - Singular Value Decomp - Normal
# * Split a the mxn matrix into 3 (Rotation, recaling (+/-), rotation)
# * Expressed as M = U * Sigma * V^T
# *     U (mxm) -> Orthonormal columns
# *     Sigma (mxn) -> Diagonal values
# *     V (nxn) -> Orthonomal columns
# * It wouldn't be machine learning if there was no optimisation problem -> In this case it would be matrix completion.
# TODO - Consider that you may get more than one table

# ? - Σ is a diagonal matrix containing the singular values in descending order.
u, sigma, v_T = svds(r_centered, k=50)  # Limit to the top 50 components
sigma = np.diag(sigma)
matrix_rank(r_centered)  # * To assess the max. number of latent vectors.

# ? - Escogemos el número optimo de latent spaces para el minimo error.
latents = [3, 10, 20, 30, 40, 50, 150, 300]
rmse_errors = []
for latent_dim in latents:
    U, sigma, v_T = svds(r_centered, k=latent_dim)
    sigma = np.diag(sigma)  # ? Force it to be diagonal
    # ? - Reconstruction of the matrix y_app = U*Sigma*Vt + user_mean (We could also add the item mean)
    r_pred = np.dot(np.dot(U, sigma), v_T) + user_ratings_mean.reshape(-1, 1)
    r_pred[r_pred < 0] = 0
    r_pred[r_pred > 5] = 5
    mse = np.square(np.subtract(r, r_pred)).mean  # ? - MSE formula
    rmse = math.sqrt(mse)
    rmse_errors.append(rmse)  # * Add it to the list

# ? - Plot opcional para observar como progresa el SVD
# ? - Hacemos ahí un elbow method supongo.
# TODO: Determinar si se debe de escoger un threshold
plt.xlabel("Latent Dimension")
plt.ylabel("RMSE")
plt.plot(latents, rmse_errors, "o-")
plt.show()

r_pred_df = pd.DataFrame(r_pred)
r_pred.head()

# * Ways to get all of the recommendations at once by constructing the full matrix
# Y = U Sigma Vt + mean
all_user_predicted_ratings = np.dot(np.dot(u, sigma), v_T) + user_ratings_mean.reshape(
    -1, 1
)

all_user_predicted_ratings[all_user_predicted_ratings < 0] = 0
all_user_predicted_ratings[all_user_predicted_ratings > 5] = 5

preds_df = pd.DataFrame(all_user_predicted_ratings, columns=r_df.columns)
preds_df.head()


# * Function to recommend movies per user.
# ? - How it works is returning the movies with the highest predicted rating.
# ? - Consider only the user rating, no title, no genre.
# TODO - How would we consider more information/other tables.
def recommend_movies(
    preds_df, userID, movies_df, original_ratings_df, num_recommendations=5
):
    # ? Get the user and their predictions
    user_row_num = userID - 1
    sorted_user_predictions = preds_df.iloc[user_row_num].sort_values(ascending=False)

    # ? Get the user's data and merge in the movie info to display: 
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = user_data.merge(
        movies_df, how='left', left_on='movieId', right_on='movieId'
    ).sort_values(['rating'], ascending = False)

    print(f"User {userID} has already rated {user_full.shape[0]} movies.")
    print(
        f"Recommending highest {num_recommendations} predicted ratings movies not already rated."
    )

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (
        # TODO -  What are we doing here?
        movies_df[~movies_df["movieId"].isin(user_full["movieId"])]
        .merge(
            pd.DataFrame(sorted_user_predictions).reset_index(),
            how="left",
            left_on="movieId",
            right_on="movieId",
        )
        .rename(columns={user_row_num: "Predictions"})
        .sort_values("Predictions", ascending=False)
        .iloc[:num_recommendations, :-1]
    )

    return user_full, recommendations

#! - Singular Value Decomp - Cornac


# ? - Examinar el nivel de error para el número de latentes en la matriz
# ! - Matrix Factorisation
# * Factorises the ratings matrix into the product of two lower-rank matrices. "Capturing the low-rank structure of the user-item interactions".
# * Y (mxn) => P (mxk) & Q (nxk), where k << m, n is the latent factor size. So Y approx PQ^T
# * P is the user matrix (m -> # of users) -> Rows measure user interest in item chars.
# * Q is the item matrix (n -> # of items) -> Rows measure item characteristics set.
