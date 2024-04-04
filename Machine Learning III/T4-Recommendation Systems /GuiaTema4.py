
# * Librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# * Import file 
df_path = "df_path"
df = pd.read_csv(df_path, header=0)

# * EDA
df.head()
df.info()
df.shape()
column_unique_values = df['column'].unique()

# ? - Agrupación de columnas: 
group_by_and_count = pd.DataFrame(df.groupby("ProductId")["Rating"].count())
sorted_values_by_criteria = group_by_and_count.sort_values("Rating", ascending=False)
sorted_values_by_criteria.head(10)

# * Take a sample of the dataset
df_sample = df.sample(n=10000, random_state=42)


# ! Recommender systems: 
# ? - Collaborative Filtering <- WE ARE INTERESTED IN THIS ONE 
    # * - Model-based filtering technique <- Clustering, NN, Association, etc... 
        # ? Use ML to find user ratings of unrated items: PCA SVD, Neural nets 
        # ? Performance reduction with sparse data. 
    # * - Memory-based filtering technique  <- User Based / Item Based <- THIS ONE FOR THE PRACTICAL 
        # ? Based on cosine similarity or pearson correlation and taking the avg. of ratings. 
        # ? Non-scalable for sparse data. 
        #TODO: User-based -> Ratings provided to similar users to A, to make recommendations to A. Weighted average. 
        #TODO: Item-based -> To make recommendations for target item B, determine a set of items similar to B. 

        # ! - Differences: user-based in using the ratings of neighboring users. Item-based predicted using the user's own ratings on neighbouring items. 
        # ? - Awesome, but what becomes a similar product then? The problem principally is finding the top-k items. 

        
# ? - Content-Based Filtering 


# ! - Example I: User-Based Filtering 
# * https://medium.com/@corymaklin/memory-based-collaborative-filtering-user-based-42b2679c6fb5
# As an example, we're going to take the movies dataset
#TODO: We're not going to get a test dataset in the exam, so what do we do there?
train_df = pd.read_csv('ml-100k/u1.base', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
test_df = pd.read_csv('ml-100k/u1.test', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

train_df.head()

# * Construct the ratings matrix: 
# ? - values -> Cell values | index -> Rows | columns -> Columns
ratings_matrix = pd.pivot_table(train_df, values='rating', index= 'user_id', columns='item_id')

# * Normalise the ratings matrix by subtracting every user's rating by the mean users rating: 
normalized_ratings_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis = 1), axis = 0)

# ? - Case 1. Pearson Correlation to determine similarity 
similarity_matrix = ratings_matrix.T.corr() # * This determines the similarity of each user 

# ? - Case 2. Cosine similarity (WE WOULD NOW HAVE TO IMPUTE THE MISSING DATA -> Most common method: Fill in with the user or item average rating)
# ?           We can proceed with this as long as all of the items have been normalised first. 
# ? - If we want to fill it in with zeroes: 
item_similarity_cosine = cosine_similarity(normalized_ratings_matrix.fillna(0))
item_similarity_cosine = cosine_similarity(normalized_ratings_matrix.fillna(ratings_matrix.T.mean()[user_id]))
item_similarity_cosine = cosine_similarity(normalized_ratings_matrix.fillna(ratings_matrix.T.mean()[item_id]))
item_similarity_cosine

# * Calculate the score according to the formula: 
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
    if item_id not in ratings_matrix.columns: #
        return 2.5 
    
    similarity_scores = similarity_matrix[user_id].drop(labels=user_id) # ? Take out the user itself, so that it doesn't self-match 
    normalized_ratings = normalized_ratings_matrix[item_id].drop(index = user_id) # ? For all of the ratings, take out the user itself 

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
    return avg_user_rating + total_score/total_weight # ? Adding back the original weight 

# * Score testing 
test_ratings = np.array(test_df["rating"])
user_item_pairs = zip(test_df["user_id"], test_df["item_id"])
pred_ratings = np.array([calculate_score(user_id, item_id) for (user_id, item_id) in user_item_pairs])
print(np.sqrt(mean_squared_error(test_ratings, pred_ratings)))


# * Baseline rating - Whic is just the mean of all ratings given
baseline_rating = train_df["rating"].mean()
baseline_ratings = np.array([baseline_rating for _ in range(test_df.shape[0])])
print(np.sqrt(mean_squared_error(test_ratings, baseline_ratings)))

# ! - Example 2: Item-based filtering 