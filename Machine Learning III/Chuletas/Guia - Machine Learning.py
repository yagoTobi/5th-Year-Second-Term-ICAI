# * ################################################################################################
# * ##############     Handy Guide ICAI - MACHINE LEARNING III   - Yago Tobio Souto  ###############
# * ################################################################################################

# * Librerias
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import cm

import matplotlib as mpl
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


import os
import sys
import time
import math
import torch
import cornac
import warnings
import operator
import itertools
import scipy.stats
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import seaborn as sns
import tensorflow as tf
import scipy.sparse as sp
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data

from tabulate import tabulate
from cornac.utils import cache
from sklearn.manifold import TSNE
from adjustText import adjust_text
from collections import defaultdict
from scipy.sparse.linalg import svds
from torch.utils.data import Dataset
from numpy.linalg import matrix_rank
from cornac.datasets import movielens
from matplotlib.cbook import boxplot_stats
from cornac.eval_methods import RatioSplit

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.constants import SEED
from sklearn.metrics import mean_squared_error
from elasticsearch import Elasticsearch, helpers
from sklearn.metrics.pairwise import cosine_similarity
from cornac.models import MF, NMF, BaselineOnly, BPR, WMF, UserKNN, ItemKNN, SVD

from recommenders.utils.notebook_utils import store_metadata
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

print(f"System version: {sys.version}")
print(f"Cornac version: {cornac.__version__}")
print(f"Tensorflow version: {tf.__version__}")

SEED = 42
VERBOSE = False

# * #######################################################################################
# * #######################################################################################
# * #######################################################################################
# * ### Indice                                                                         ####
# * ###                                                                                ####
# * ### 0. EDA                                     (Linea 86)                          ####
# * ### 1. PCA                                                                         ####
# * ### 2. Clustering                                                                  ####
# * ### 3. Memory-Based Filtering                  (Linea 281)                         ####
# * ###    a. User-Based Filtering                 (Linea 281)                         ####
# * ###    b. Item-Based Filtering                 (Linea 400)                         ####
# * ### 4. Model-Based Collaborative Filtering     (Linea 498)                         ####
# * ###    a. Singular Value Decomposition         (Linea 526)                         ####
# * ###    b. Matrix Factorisation                 (Linea 652)                         ####
# * ### 5. Implicit Feedback                       (Linea 880)                         ####
# * #######################################################################################
# * #######################################################################################
# * #######################################################################################

# * Import file
df_path = "df_path"
df = pd.read_csv(df_path, header=0)

#!##########################################################################################
#!###################         EDA               ############################################
#!##########################################################################################

# * - Preview, Resumen de caracteristicas, num filas x columns.
df.head()
df.info()
df.shape()

# * - Eliminar columnas
df.drop("Timestamp", axis=1, inplace=True)

# * - Ubicar los na's
df.isna().sum()

# * - Coger una muestra aleatoria de los datos
df_sample = df.sample(n=10000, random_state=42)

# ? - Hallar el número de ratings individuales
# ? - (Útil para observar proporción y escala)
df.value_counts("Rating", normalize=True)

column_unique_values = df_sample[
    "column"
].unique()  # ? - Esto obtiene los valores únicos por col

number_column_unique_values = df_sample[
    "column"
].nunique()  # ? - Esto el número de valores únicos

# * - Agrupación de columnas:
group_by_and_count = pd.DataFrame(df_sample.groupby("ProductId")["Rating"].count())
sorted_values_by_criteria = group_by_and_count.sort_values("Rating", ascending=False)
sorted_values_by_criteria.head(10)

# TODO: Añadir table merging

# * - Obten en array los tipos de datos por columna:
data_types = [str(df_sample[column].dtype) for column in df_sample.columns]

# * - Meter info previa en un dataFrame:
unique_counts = df_sample.nunique()
unique_values = [df_sample[column].unique() for column in df_sample.columns]
data_types = [str(df_sample[column].dtype) for column in df_sample.columns]
unique_counts_df = pd.DataFrame(
    {
        "feature": df_sample.columns,
        "unique_count": unique_counts,
        "unique_values": unique_values,
        "data_type": data_types,
    }
)
unique_counts_df


# * Función de análisis de outliers de ratings:
def explore_outliers(df, num_vars):
    """
    Explora y identifica los valores atípicos de variables numéricas en un DataFrame.

    Retorna:
    - outliers_df (diccionario): Diccionario con las variables numéricas como claves. Cada valor es otro diccionario
      con las claves 'values' (valores atípicos), 'positions' (posiciones de los valores atípicos en el DataFrame)
      e 'indices' (índices de los valores atípicos en el DataFrame).
    """
    outliers_df = dict()
    for k in range(len(num_vars)):
        var = num_vars[k]
        sns.boxplot(df, x=var)
        outliers_df[var] = boxplot_stats(df[var])[0][
            "fliers"
        ]  # ? - Boxplot de TODOS LOS RATINGS EN NUESTRA MUESTRA
        out_pos = np.where(df[var].isin(outliers_df[var]))[0].tolist()
        out_idx = [df[var].index.tolist()[k] for k in out_pos]
        outliers_df[var] = {
            "values": outliers_df[var],
            "positions": out_pos,
            "indices": out_idx,
        }
    return outliers_df


# * Obtener los outliers y visualizar el boxplot.
outlier_ratings = explore_outliers(df_sample, ["Rating"])
# * Obtener porcentaje de outliers de nuestra muestra:
print(
    "Percentage of outliers:",
    round(len(outlier_ratings.get("Rating").get("indices")) / len(df_sample), 3) * 100,
    "%",
)

# ! - Si hay un sesgo muy claro en el boxplot, NO recomendamos quitar las anomalías para
# ! - capturar todos los comportamientos posibles de usuarios.
# * En caso de querer quitar los outliers:
df_sample.drop(outlier_ratings.get("Rating").get("indices"), inplace=True)


# * - Calculo de Sparsity. Nos dice si nuestro dataset exhibe propiedades long-tail
# * - Si el sparsity es alto, yo optaría por hacer Cosine similarity
# * - Como de llena esta nuestra matriz de ratings:
def print_sparsity(df):
    n_users = df.UserId.nunique()
    n_items = df.ProductId.nunique()
    n_ratings = len(df)
    rating_matrix_size = n_users * n_items
    sparsity = 1 - n_ratings / rating_matrix_size

    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    print(f"Number of available ratings: {n_ratings}")
    print(f"Number of all possible ratings: {rating_matrix_size}")
    print("-" * 40)
    print(f"SPARSITY: {sparsity * 100.0:.2f}%")


print_sparsity(df_sample)

# * - Obten los productos con mayor número de críticas
item_rate_count = (
    df_sample.groupby("ProductId")["UserId"].nunique().sort_values(ascending=False)
)
item_rate_count  # ? - Get the number of reviews for a product

# * - Opcional: Plot para observar si hay long tail property: (CUIDADO CON NOMBRES DE COLS)
popular_products = pd.DataFrame(df_sample.groupby("ProductId")["Rating"].count())
most_popular = popular_products.sort_values("Rating", ascending=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
# First plot
axes[0].bar(
    x=range(len(item_rate_count)),
    height=item_rate_count.values,
    width=5.0,
    align="edge",
)
axes[0].set_xticks([])
axes[0].set(
    title="Long tail of rating frequency",
    xlabel="Item ordered by decreasing frequency",
    ylabel="#Ratings",
)

# Second plot adaptation
# Assuming most_popular is a Series. If it's a DataFrame, you might need to adjust this part.
x_pos = range(len(most_popular.head(30)))  # Generate x positions
axes[1].bar(x=x_pos, height=most_popular.head(30)["Rating"], align="center")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(most_popular.head(30).index, rotation="vertical")
axes[1].set(
    title="Top 30 Most Popular Items",
    xlabel="Item",
    ylabel="Frequency or some other metric",
)

plt.tight_layout()
plt.show()


"""
! DE ESTE PUNTO EN ADELANTE - EDA PARA RECOMMENDATION SYSTEMS: 

! IMPORTANTE, ESTAMOS ASUMIENDO QUE EN CADA TABLA HAY: userId, itemId, rating, timestamp
! Asegurate que tienes una chuleta de pandas para eliminar las columnas 
! Tambien nos podríamos enfrentar con tablas que tengan mas datos que puedan ser utiles. 
! Pero por ahora el foco de la asignatura esta con el user-item matrix. No por ejemplo movie genre.
"""

# * Generar la matriz de ratings:
ratings_matrix = df_sample.pivot_table(
    index="UserId",
    columns="ProductId",
    values="Rating",
)

# * EDA de la matriz de ratings:
ratings_matrix.head()
df = ratings_matrix
df["Mean Rating"] = df.mean(axis=1)  # ? - Get the mean score for each user
sns.histplot(
    x="Mean Rating", binwidth=0.5, data=df
)  # ? - Histograma de la media de puntuación

# * PARA PODER HACER ITEM Y USER PROFILING
# * User-profiling para User-based
# ? - Dataset para agrupar los items
df_user_10k = pd.read_csv("path.csv").set_index("UserId").drop("Timestamp", axis=1)
items = df_user_10k.groupby(
    "ProductId"
)  # ? - Obtener lista de productos criticados por usuario
items.get_group("B002OVV7F0")  # ? - Pass ProductId - Get the ratings

# ? - Dataset para agrupar los users
df_item_10k = pd.read_csv("path.csv").set_index("ProductId").drop("Timestamp", axis=1)
users = df_item_10k.groupby("UserId")  # ? - Obtener lista de usuarios por producto
users.get_group("A39HTATAQ9V7YF")  # ? - Pass UserId - Get the ratings for a user

# ? - Observar distribución ratings producto especifico:
df_item_10k.loc["B0050QLE4U"].hist()
# ? - Observar distribución ratings user especifico:
df_user_10k.loc["A39HTATAQ9V7YF"].hist()

#!##################################################################################################################################
#!########## Principal Component Analysis             #############################################################################
#!##################################################################################################################################

# * - Paso I: Copiar el dataset con el fin de hacer operaciones + define columnas
df_pca = df.copy()
df_pca.head()

CATEGORICAL_COLUMNS = [
    "TIME_OF_DAY",
    "AIRCRAFT",
    "AC_MASS",
    "PHASE_OF_FLIGHT",
    "SPECIES",
    "NUM_STRUCK",
]

NUMERICAL_COLUMNS = ["INCIDENT_YEAR", "HEIGHT", "SPEED", "DISTANCE", "COST_INFL_ADJ"]

# * - Paso II: Solo aplicar el PCA a las variables numericas. Convertir variables si necesario
# * - Despues arrancamos haciendo el train-test split:
X = df_pca[NUMERICAL_COLUMNS]
X = X.drop(columns=["COST_INFL_ADJ"])  # * Variable target a identificar
y = df_pca.COST_INFL_ADJ  # * Continuous data for the darget
X_train, X_test, y_train, y_tests = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# * - Paso III: Aplicamos el Standard Scaler con el objetivo de normalizar
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reconstruction error on MNIST vs number of latent dimensions used by PCA
# * This is dimension D
X_rank = np.linalg.matrix_rank(X_train)

# * Generate the K's vector - To test out for the optimal L, number of latent dimensions.
Ks = [1, 2, 3, 4, 5, 6, 7, ...]

# ? - Vectors which are intended to store the RMSE values for train and test datasets.
RMSE_train = np.arange(len(Ks))
RMSE_test = np.arange(len(Ks))

# * For each L which we want to test out perform a PCA and it's corresponding reconstruction
# * Both for the train & test datasets - Then for that specific L, log what the RMS error is in order to plot it.
for index, K in enumerate(Ks):
    pca = PCA(n_components=K)

    Xtrain_transformed = pca.fit_transform(X_train)
    Xtrain_proj = pca.inverse_transform(Xtrain_transformed)
    RMSE_train[index] = mean_squared_error(X_train, Xtrain_proj, squared=False)

    Xtest_transformed = pca.transform(X_test)
    Xtest_proj = pca.inverse_transform(Xtest_transformed)
    RMSE_test[index] = mean_squared_error(X_test, Xtest_proj, squared=False)

# * Plot to find the reconstruction errors:  borth for train and test.
# Asumiendo que X_train y X_test están definidos
max_components = min(X_train.shape[0], X_train.shape[1]) - 1
Ks = list(
    range(1, max_components + 1)
)  # Ajustando dinámicamente el rango basado en el tamaño del conjunto de datos

RMSE_train = np.zeros(len(Ks), dtype=float)
RMSE_test = np.zeros(len(Ks), dtype=float)

for index, K in enumerate(Ks):
    pca = PCA(n_components=K)
    Xtrain_transformed = pca.fit_transform(X_train)
    Xtrain_proj = pca.inverse_transform(Xtrain_transformed)
    RMSE_train[index] = mean_squared_error(X_train, Xtrain_proj, squared=False)

    Xtest_transformed = pca.transform(X_test)
    Xtest_proj = pca.inverse_transform(Xtest_transformed)
    RMSE_test[index] = mean_squared_error(X_test, Xtest_proj, squared=False)

# Configurar la figura y los ejes para dos subplots: uno al lado del otro
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 fila, 2 columnas

# Gráfico para el conjunto de entrenamiento
axs[0].plot(Ks, RMSE_train, marker="o", color="blue")
axs[0].set_title("Train Set Reconstruction Error")
axs[0].set_xlabel("Number of Principal Components")
axs[0].set_ylabel("RMSE")

# Gráfico para el conjunto de prueba
axs[1].plot(Ks, RMSE_test, marker="x", color="red")
axs[1].set_title("Test Set Reconstruction Error")
axs[1].set_xlabel("Number of Principal Components")
# axs[1].set_ylabel("RMSE")  # Opcional, ya que comparten el mismo eje Y

plt.tight_layout()  # Ajustar automáticamente los parámetros de la subtrama para dar un relleno especificado
plt.show()

#!##################################################################################################################################
#!########## Memory-Based: User-based filtering Cornac #############################################################################
#!##################################################################################################################################

# train_df = pd.read_csv(
#    "ml-100k/u1.base",
#    sep="\t",
#    header=None,
#    names=["user_id", "item_id", "rating", "timestamp"],
# )
#
# test_df = pd.read_csv(
#    "ml-100k/u1.test",
#    sep="\t",
#    header=None,
#    names=["user_id", "item_id", "rating", "timestamp"],
# )
#
# train_df.head()

# * Normalise the ratings matrix by subtracting every user's rating by the mean users rating:
normalized_ratings_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis=1), axis=0)


# * Generación de modelos user-based con Pearson, Cosine y centrados
def userknn_cornac(df: pd.DataFrame):

    df = df.astype({"UserId": object, "ProductId": object})
    records = df.to_records(index=False)
    result = list(records)

    K = 3  # number of nearest neighbors
    VERBOSE = False
    SEED = 42
    uknn_cosine = UserKNN(
        k=K, similarity="cosine", name="UserKNN-Cosine", verbose=VERBOSE
    )
    uknn_cosine_mc = UserKNN(
        k=K,
        similarity="cosine",
        mean_centered=True,
        name="UserKNN-Cosine-MC",
        verbose=VERBOSE,
    )
    uknn_pearson = UserKNN(
        k=K, similarity="pearson", name="UserKNN-Pearson", verbose=VERBOSE
    )
    uknn_pearson_mc = UserKNN(
        k=K,
        similarity="pearson",
        mean_centered=True,
        name="UserKNN-Pearson-MC",
        verbose=VERBOSE,
    )

    # Metrics
    rec_300 = cornac.metrics.Recall(k=300)
    rec_900 = cornac.metrics.Recall(k=900)
    prec_30 = cornac.metrics.Precision(k=30)
    rmse = cornac.metrics.RMSE()
    mae = cornac.metrics.MAE()

    ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
    cornac.Experiment(
        eval_method=ratio_split,
        models=[uknn_cosine, uknn_cosine_mc, uknn_pearson, uknn_pearson_mc],
        metrics=[rec_300, rec_900, prec_30, rmse, mae],
    ).run()

    userknn_models = {
        "uknn_cosine": uknn_cosine,
        "uknn_cosine_mc": uknn_cosine_mc,
        "uknn_pearson": uknn_pearson,
        "uknn_pearson_mc": uknn_pearson_mc,
    }

    return userknn_models


userknn_models = userknn_cornac(df_sample)  # ? - Returns the data with the Metrics
# ?^^Luego tendras que justificar que modelo eliges. Esta bien que cojamos el mean centered (mc)


# * Función para hacer el perfil del usuario basado en su ID, el df de usuarios, y el modelo user-based (Pearson o cosine seleccionado)
def user_profiling(UID, model, user_df, TOPK=5):

    rating_mat = model.train_set.matrix

    UIDX = list(model.train_set.uid_map.items())[UID][0]

    print(f"UserID = {UIDX}")
    print("-" * 35)
    print(user_df.loc[UIDX])

    ratings = pd.DataFrame(rating_mat.toarray())
    user_ratings = ratings.loc[UID]
    top_rated_items = np.argsort(user_ratings)[-TOPK:]
    print(f"\nTOP {TOPK} RATED ITEMS BY USER {UID}:")
    print("-" * 35)
    print(user_df.iloc[top_rated_items.array])


# * Predicción de score para cualquier producto:
def uknn_get_scores(UID, model, user_df, TOPK=5):

    UIDX = list(model.train_set.uid_map.items())[UID][0]
    recommendations, scores = model.rank(UID)
    print(f"\nTOP {TOPK} RECOMMENDATIONS FOR USER {UIDX}:")
    print("Scores:", scores[recommendations[:TOPK]])
    print(user_df.iloc[recommendations[:TOPK]])


uknn_get_scores(2, model, df_user_10k)

#!###################################################################################################################################
#!########## Memory-Based: Item-based filtering - Cornac ############################################################################
#!###################################################################################################################################

# * Predict user's rating for one movie process:
# * 1. Create a list of the movies which the user 1 has watched and rated.
# * 2. Rank the similarities between the movies that user 1 has rated and the movie to predict.
# * 3. Select top n movies with the highest similarity scores.
# * 4. Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1.


# * Función para generar los modelos item-based utilizando Pearson, Cosine, etc... ajustados o no
def itemknn_cornac(df):
    df = df.astype({"UserId": object, "ProductId": object})
    records = df.to_records(index=False)
    result = list(records)
    K = 50  # number of nearest neighbors
    VERBOSE = False
    SEED = 42
    iknn_cosine = ItemKNN(
        k=K, similarity="cosine", name="ItemKNN-Cosine", verbose=VERBOSE
    )
    iknn_cosine_mc = ItemKNN(
        k=K,
        similarity="cosine",
        mean_centered=True,
        name="ItemKNN-Cosine-MC",
        verbose=VERBOSE,
    )
    iknn_pearson = ItemKNN(
        k=K, similarity="pearson", name="ItemKNN-Pearson", verbose=VERBOSE
    )
    iknn_pearson_mc = ItemKNN(
        k=K,
        similarity="pearson",
        mean_centered=True,
        name="ItemKNN-Pearson-MC",
        verbose=VERBOSE,
    )

    # Metrics
    rmse = cornac.metrics.RMSE()
    mae = cornac.metrics.MAE()

    ratio_split = RatioSplit(result, test_size=0.2, seed=SEED, verbose=VERBOSE)
    cornac.Experiment(
        eval_method=ratio_split,
        models=[iknn_cosine, iknn_pearson, iknn_pearson_mc, iknn_cosine_mc],
        metrics=[rmse, mae],
    ).run()
    itemknn_models = {
        "iknn_cosine": iknn_cosine,
        "iknn_pearson": iknn_pearson,
        "iknn_pearson_mc": iknn_pearson_mc,
        "iknn_cosine_mc": iknn_cosine_mc,
    }
    return itemknn_models


itemknn_models = itemknn_cornac(df_sample)


# * Función para hacer el profiling de los items en base al modelo seleccionado y el dataset. Top K.
def item_profiling(UID, model, item_df, TOPK=5):

    rating_mat = model.train_set.matrix

    UIDX = list(model.train_set.iid_map.items())[UID][0]

    print(f"ProductID = {UIDX}")
    print("-" * 35)
    print(item_df.loc[UIDX])

    ratings = pd.DataFrame(rating_mat.toarray())
    item_ratings = ratings.iloc[UID]
    top_rated_items = np.argsort(item_ratings)[-TOPK:]
    print(f"\nTOP {TOPK} RECOMMENDED USERS FOR ITEM {UID}:")
    print("-" * 35)
    print(df_item_10k.iloc[top_rated_items.array])


model = itemknn_models.get("iknn_cosine_mc")
top_rated_items = item_profiling(2, model, df_item_10k)


# * En caso que quieras una función más especifica a los ratings: (Puedes pasar)
def itemknn_get_scores(UID, model, item_df, TOPK=5):

    UIDX = list(model.train_set.iid_map.items())[UID][0]
    recommendations, scores = model.rank(UID)
    print(f"\nTOP {5} USERS FOR ITEM {UIDX}:")
    print("Scores:", scores[recommendations[:TOPK]])
    print(item_df.iloc[recommendations[:TOPK]])


# * Indice del item, modelo seleccionado, y le pasas el dataset
itemknn_get_scores(3, model, df_item_10k)

#!##################################################################################################################################
#!###### Model-based collaborative filtering techniques ############################################################################
#!##################################################################################################################################

# * Asegurate que tienes la matriz del pivotTable de antemano (ratings_matrix)
# * En caso que no la tengas:
#   ratings_matrix = df_sample.pivot_table(
#       index="UserId",
#       columns="ProductId",
#       values="Rating",
#    )

# * Rellenamos la ratings matrix con la media
mean_rating = 2.5
r_df = ratings_matrix.fillna(mean_rating)

# * Pasamos el df a numpy
r = r_df.to_numpy()

# * - Centramos los ratings al subtraer la media general de la matriz.
user_ratings_mean = np.mean(r, axis=1)
r_centered = r - user_ratings_mean.reshape(-1, 1)

# * - Verificamos que la dimension se ha mantenido al subtraer
# * - Multiplica los dos numeros del r.shape y ver si coincide con el count_nonzero
print(r.shape)
print(np.count_nonzero(r))

#!##################################################################################################################################
#!################ Singular Value Decomp - CORNAC ##################################################################################
#!##################################################################################################################################


# * - Función para generar el módelo SVD
def svd_cornac(df, k_min=10, k_max=2000, step=100):
    """
    Ejecuta experimentos SVD para un rango de valores de 'k' sobre un DataFrame utilizando Cornac.

    Devuelve:
    - (lista, cornac.Experiment): Lista de modelos SVD entrenados y el objeto experimento con los resultados.

    """
    df = df.astype({"UserId": object, "ProductId": object})
    records = df.to_records(index=False)
    result = list(records)

    VERBOSE = False
    SEED = 42

    svd_models = []
    k_values = np.arange(k_min, k_max, step)
    for k in k_values:
        svd_models.append(
            SVD(
                name="SVD" + str(k),
                k=k,
                max_iter=30,
                learning_rate=0.01,
                lambda_reg=0.02,
                verbose=True,
            )
        )

    # Metrics
    rmse = cornac.metrics.RMSE()
    mae = cornac.metrics.MAE()

    ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
    svd_experiment = cornac.Experiment(
        eval_method=ratio_split,
        models=svd_models,
        show_validation=True,
        metrics=[rmse, mae],
    )
    svd_experiment.run()

    return svd_models, svd_experiment


svd_models, svd_experiment = svd_cornac(df_sample, 100, 500, 50)


# * Función para hacer el plot comparando el número de dimensiones latentes vs. Error RMSE.
# * Propósito: Obtener el K óptimo
def plot_rmse_cornac(experiment, metric_name="RMSE"):
    metric_values = []
    names_models = []
    for i in range(len(experiment.result)):
        metric_values.append(
            svd_experiment.result[i].metric_avg_results.get(metric_name)
        )
        names_models.append(svd_experiment.result[i].model_name)

    plt.xlabel("Latent Dimensions")
    plt.ylabel("RMSE")
    plt.title("SVD")
    plt.plot(names_models, metric_values, "o-")
    plt.show()


plot_rmse_cornac(svd_experiment)

# * En caso de querer hacer un baseline model para SVD - Solo considerando los bias:
df = df_sample.astype({"UserId": object, "ProductId": object})
records = df.to_records(index=False)
result = list(records)

# ? - Instantiate an evaluation method to split data into train and test sets.
ratio_split = cornac.eval_methods.RatioSplit(
    data=df_sample.values, test_size=0.1, verbose=True
)

# ? - Instantiate the models of interest
bo = cornac.models.BaselineOnly(
    max_iter=30, learning_rate=0.01, lambda_reg=0.02, verbose=True
)

# Instantiate evaluation measures
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()

# Instantiate and run an experiment.
cornac.Experiment(eval_method=ratio_split, models=[bo], metrics=[mae, rmse]).run()
# ? - Tras ejecutar esto compara con el bloque de arriba


# * Función para recomendar a un usuario mediante el indice su producto
def recommend_products(index, model, data, num_products=5):

    print(
        "Name of Model:", svd_models[n].name
    )  # ? - Sustituir el n, por el indice del modelo con menor RMSE

    # Rank all test items for a given user.
    df_rank = pd.DataFrame(
        {"ranked_items": model.rank(index)[0], "item_scores": model.rank(index)[1]},
        columns=["ranked_items", "item_scores"],
    )
    print(
        "Target UserId", df_sample.iloc[index].UserId
    )  # * Cuidado aqui con el df_smaple

    df_rank.sort_values("item_scores", ascending=False, inplace=True)

    print(
        "Recommended products:",
        data.iloc[df_rank.head(num_products).ranked_items.values]["ProductId"].values,
    )
    print("Predicted scoreds: ", df_rank.head(num_products).item_scores.values)


recommend_products(1, svd_models[n], df_sample)


#!##################################################################################################################################
#!########################## Matrix Factorisation ##################################################################################
#!##################################################################################################################################

# * Factorises the ratings matrix into the product of two lower-rank matrices. "Capturing the low-rank structure of the user-item interactions".
# * Y (mxn) => P (mxk) & Q^T (kxn), where k << m, n is the latent factor size. So Y^ = PQ^T
# * P is the user matrix (m -> # of users) -> Rows measure user interest in item chars.
# * Q is the item matrix (n -> # of items) -> Rows measure item characteristics set.

#! Option 1 - SINGLE MF model no-comparison:
# * Load the data for the MF
# ? Option 1 - Non-csv
data = movielens.load_feedback(variant="100K")
# ? !!! - Option 2 - CORNAC CSV IMPORT
pandas_df = pd.read_csv("csv_path")
data = cornac.data.Dataset.from_uir(pandas_df.itertuples(index=False))

# * Data split and calculate the RMSE
rs = RatioSplit(data, test_size=0.2, seed=SEED, verbose=VERBOSE)
rmse = cornac.metrics.RMSE()

K = 10
lbd = 0.01  # ? Lambda -> Regularisation
mf = MF(
    k=K,
    max_iter=20,
    learning_rate=0.01,
    lambda_reg=lbd,
    use_bias=False,
    verbose=VERBOSE,
    seed=SEED,
    name=f"MF(K={K},lambda={lbd:.4f})",
)

# * Execute the MF model
cornac.Experiment(eval_method=rs, models=[mf], metrics=[rmse]).run()
# ? - ^^If you only have one model.

# .fit(dataset) <- Attach to the mf function if you want to execute the below section.
# ^^Also don't forget about the dataset
# print("User factors:\n", mf.u_factors)
# print("Item factors:\n", mf.i_factors)


#! Option 2 - Multiple models includes Non-Negative Matrix Factorisation - THIS ONE :):

# * NMF: Variant where the latent factors are constrained to be non-negative
# * Ideal for non-negative factors like image processing, text mining, and rec. systems.
# * As there are no negative factors.
# * Allows for better interpretabiliy to reason with positive values:


def mf_cornac(df, K=10):
    df = df.astype({"UserId": object, "ProductId": object})
    records = df.to_records(index=False)
    result = list(records)
    VERBOSE = False
    SEED = 42
    lbd = 0.01
    baseline = BaselineOnly(
        max_iter=20, learning_rate=0.01, lambda_reg=lbd, verbose=VERBOSE
    )
    mf1 = MF(
        k=K,
        max_iter=20,
        learning_rate=0.01,
        lambda_reg=0.0,
        use_bias=False,
        verbose=VERBOSE,
        seed=SEED,
        name=f"MF(K={K})",
    )
    mf2 = MF(
        k=K,
        max_iter=20,
        learning_rate=0.01,
        lambda_reg=lbd,
        use_bias=False,
        verbose=VERBOSE,
        seed=SEED,
        name=f"MF(K={K},lambda={lbd:.4f})",
    )
    mf3 = MF(
        k=K,
        max_iter=20,
        learning_rate=0.01,
        lambda_reg=lbd,
        use_bias=True,
        verbose=VERBOSE,
        seed=SEED,
        name=f"MF(K={K},bias)",
    )
    nmf = NMF(
        k=K,
        max_iter=200,
        learning_rate=0.01,
        use_bias=False,
        verbose=VERBOSE,
        seed=SEED,
        name=f"NMF(K={K})",
    )
    ratio_split = RatioSplit(result, test_size=0.1, seed=SEED, verbose=VERBOSE)
    cornac.Experiment(
        eval_method=ratio_split,
        models=[baseline, mf1, mf2, mf3, nmf],
        metrics=[cornac.metrics.RMSE()],
    ).run()

    mf_models = {"baseline": baseline, "mf1": mf1, "mf2": mf2, "mf3": mf3, "nmf": nmf}
    return mf_models


K = 10
mf_models = mf_cornac(df_sample, K)

model = mf_models.get("mf3")  # ? - Selecciona el que mejor RMSE tenga
var_df = pd.DataFrame(
    {"Factor": np.arange(K), "Variance": np.var(model.i_factors, axis=0)}
)

# * Observar la info y varianza de cada dimensión latente del MF
fig, ax = plt.subplots(figsize=(12, 5))
plt.title("MF")
sns.barplot(x="Factor", y="Variance", data=var_df, hue="Factor", legend=False, ax=ax)


# * - Codigo para visualizar en 2D el espacio latente. A ver si hay agrupaciones
# TOP2F = (0, 2)
# SAMPLE_SIZE = 500
#
# mf = model
# rng = np.random.RandomState(SEED)
# sample_inds = rng.choice(
#    np.arange(mf.i_factors.shape[0]), size=SAMPLE_SIZE, replace=False
# )
# = Selecting the top 2 latent vectors and seeing how it's grouping up the films
# sample_df = pd.DataFrame(data=mf.i_factors[sample_inds][:, TOP2F], columns=["x", "y"])
#
# sns.lmplot(x="x", y="y", data=sample_df, height=11.0, fit_reg=False)
# item_idx2id = list(mf.train_set.item_ids)
# titles = item_df.loc[[int(item_idx2id[i]) for i in sample_inds]]["Title"].values
# adjust_text(
#    [plt.text(*sample_df.loc[i].values, titles[i], size=10) for i in range(len(titles))]
# );

# ! - Si después quieres hacer un clustering con los datos del MF, procede a ver la última parte de la solución.

# TODO: When is NMF vs. MF relevant? And how about choosing between SVD and MF?
#!##################################################################################################################################
#!########### Non-negative matrix factorisation (NMF) - Puedes pasar de esta sección ##############################################################################
#!##################################################################################################################################
k = 10
nmf = NMF(
    k=k,
    max_iter=100,  # ? - How do we decide on the number of iterations
    learning_rate=0.01,
    lambda_reg=0.0,
    verbose=VERBOSE,
    seed=SEED,
    name=f"NMF (K = {k})",
)

pandas_df = pd.read_csv("csv_path")
data = cornac.data.Dataset.from_uir(pandas_df.itertuples(index=False))

rs = RatioSplit(data, test_size=0.2, seed=SEED, verbose=VERBOSE)
rmse = cornac.metrics.RMSE()
cornac.Experiment(eval_method=rs, models=[nmf], metrics=[rmse]).run()

# ? - Visualise the variance for each latent factor in the NFM
var_df = pd.DataFrame(
    {"Factor": np.arange(K), "Variance": np.var(nmf.i_factors, axis=0)}
)
fig, ax = plt.subplots(figsize=(12, 5))
plt.title("NFM")
sns.barplot(x="Factor", y="Variance", data=var_df, palette="ch:.25", ax=ax)

# ? - Create a the reconstruction matrix based on the original dimensions
recons_matrix = pd.DataFrame(
    index=range(ratings_matrix.shape[0]), columns=range(ratings_matrix.shape[1])
)
# ? - Populate with the values
for u, i in itertools.product(
    range(recons_matrix.shape[0]), range(recons_matrix.shape[1])
):
    recons_matrix[u, i] = mf.score(u, i)
# ? - ^^Careful if you had multiple models, this is for a single one.

ratings_mask = (ratings_matrix > 0).astype(float)  # ? - To make them decimals

rmse = np.sqrt((((ratings_matrix - recons_matrix) ** 2) * ratings_mask).mean())
print(f"\nRMSE = {rmse:.3f}")
print("Reconstructed matrix:")
pd.DataFrame(
    recons_matrix.round(2),
    index=[f"User {u + 1}" for u in np.arange(df.num_users)],
    columns=[f"Item {i + 1}" for i in np.arange(df.num_items)],
)

# * - Identify the top items associated with each latent factor in an NMF
item_idx2id = list(nmf.train_set.item_ids)  # ? - Map the original id's of the items
top_items = {}
for k in range(K):  # ? - For each latent vector
    # ? - For each column in the latent matrix, pick the top five items (Slice the last 5 items in ascending order. [::-1] then just reverses it)
    top_inds = np.argsort(nmf.i_factors[:, k])[-5:][::-1]
    # * Make sure you have an item df
    # ? - Append to the dictionary the latent factor with its top 5 elements
    top_items[f"Factor {k}"] = item_df.loc[[int(item_idx2id[i]) for i in top_inds]][
        "Title"
    ].values

pd.DataFrame(top_items)

# * Attempt to extract latent vector information by sorting into genre and see if they're related:
item_idx2id = list(nmf.train_set.item_ids)
top_genres = {}
for k in range(K):
    top_inds = np.argsort(nmf.i_factors[:, k])[-100:]  # ? - Same procedure
    # ? - Make sure you have an item df
    top_items = item_df.loc[
        [int(item_idx2id[i]) for i in top_inds]
    ]  # ? - Get the top films per latent ficture
    # ? - Then drop the columns to just get the genre count.
    top_genres[f"Factor {k}"] = top_items.drop(columns=["Title", "Release Date"]).sum(
        axis=0
    )
pd.DataFrame(top_genres)
# TODO: Still don't have it clear how MF and SVD fill in the remaining elements !!!

#!##################################################################################################################################
#!#######   Implicit Feedback - Interaction based     ##############################################################################
#!##################################################################################################################################
# ? It compares pairs of items -> Item's the user has interacted with vs. items they haven't.
# ? Attempts to learn a ranking that predicts the user's preference for the interacted item over the non-interacted one.

# ? +: The user chose to interact with the item
# ? -: The user chose not to interact with the item
# ? *: Item is not specific comparison => Item is not considered specific comparison (Can't be compared with itself)
# ? ?: Unknown.

#!##################################################################################################################################
#!#######   Bayesian Probability Ratings - Cornac     ##############################################################################
#!##################################################################################################################################
# * Import the data and split into train/test
data = pd.read_csv("path_to_csv")
train, test = python_random_split(data, 0.75)
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

# print("Number of users: {}".format(train_set.num_users))
# print("Number of items: {}".format(train_set.num_items))

# * BPR Model
# ? -top k items to recommend
TOP_K = 10

# ? - Model parameters
NUM_FACTORS = 250
NUM_EPOCHS = 100

bpr = cornac.models.BPR(
    k=NUM_FACTORS,  # ? - Control the dimension of the latent space.
    max_iter=NUM_EPOCHS,  # ? - Num of iterations for SGD
    learning_rate=0.01,  # ? - Controls the step size alpha for gradient update. Small in this case
    lambda_reg=0.001,  # ? - L2 Regularisation
    verbose=True,
    seed=SEED,
).fit(
    train_set
)  # ? - In case you wish to train it directly

# * The BPR model is effectively designed for item ranking. So we should only measure performance using the ranking metrics.
with Timer() as t:
    all_predictions = predict_ranking(
        bpr, train, usercol="userID", itemcol="itemID", remove_seen=True
    )
print(f"Took {t} secondes for the prediction")

all_predictions.head()  # ? - Visualise the prediction for user ratings
bpr.rank(3)[1][
    1394
]  # ? - Get the ranking of items for user with ID 3 -> Access the second element with itemID 1394
# TODO ^^ In the above case, shouldn't we apply a mask for the prediction for it to be zero?

# * Analysis of the predictions and extract their performance matrix
k = 10
# Mean Average Precision for top k prediction items
eval_map = map(test, all_predictions, col_prediction="prediction", k=k)
# Normalized Discounted Cumulative Gain (nDCG)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction="prediction", k=k)
# precision at k (min=0, max=1)
eval_precision = precision_at_k(test, all_predictions, col_prediction="prediction", k=k)
eval_recall = recall_at_k(test, all_predictions, col_prediction="prediction", k=k)

print(
    "MAP:\t%f" % eval_map,
    "NDCG:\t%f" % eval_ndcg,
    "Precision@K:\t%f" % eval_precision,
    "Recall@K:\t%f" % eval_recall,
    sep="\n",
)
warnings.filterwarnings("ignore")

#!##################################################################################################################################
#!###############   Weighted Matrix Factorisation     ##############################################################################
#!##################################################################################################################################

K = 50
wmf = WMF(
    k=K,
    max_iter=100,
    a=1.0,
    b=0.01,
    learning_rate=0.001,
    lambda_u=0.01,
    lambda_v=0.01,
    verbose=VERBOSE,
    seed=SEED,
    name=f"WMF(K={K})",
)

eval_metrics = [
    cornac.metrics.RMSE(),
    cornac.metrics.AUC(),
    cornac.metrics.Precision(k=10),
    cornac.metrics.Recall(k=10),
    cornac.metrics.FMeasure(k=10),
    cornac.metrics.NDCG(k=[10, 20, 30]),
    cornac.metrics.MRR(),
    cornac.metrics.MAP(),
]

pandas_df = pd.read_csv("csv_path")
data = cornac.data.Dataset.from_uir(pandas_df.itertuples(index=False))
rs = RatioSplit(data, test_size=0.2, seed=SEED, verbose=VERBOSE)
cornac.Experiment(
    eval_method=rs, models=[wmf, mf], metrics=eval_metrics
).run()  # ? - This will output all of the metrics mentioned
# * Consider that MF models are strong at predicting the ratings well.
# * However, WMF models are designed to rank items, by fitting binary adoptions. (A click, a purchase, a view)
# * This is more about showing interest, rather than judging how much they will like it

#!###########################################################################################################################
#!############# Factorisation Machines (FM) #################################################################################
#!###########################################################################################################################

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    print("MPS device not found.")

# * - Get the item df
item_df = pd.read_csv("path_to_csv")
# ? - Make sure to create a column with the Id index in case that the id's don't start as 0
item_df["itemId_index"] = item_df["itemId"].astype("category").cat.codes
item_df.head()

# * - Get the user df
user_df = pd.read_csv("path_to_csv")
# ? - Remember to factorise all categorical variables !!! - Select those which are relevant
user_df["gender_index"] = user_df["gender"].astype("category").cat.codes
user_df["age_index"] = user_df["age"].astype("category").cat.codes
user_df["occupation_index"] = user_df["occupation"].astype("category").cat.codes
user_df["userId_index"] = user_df["userId"].astype("category").cat.codes
user_df.head()

# * - Get the ratings df and join it with the userId and itemId
ratings_df = pd.read_csv("path_to_csv")
ratings = ratings_df.join(item_df.set_index("itemId"), on="movieId")
ratings = ratings_df.join(user_df.set_index("userId"), on="userId")

# * - Get the feature columns to prepare for Factor Machines. !!! Don't forget to modify for the real ones.
# TODO - Is multi-fesature recommendation systems only relevant when it comes to implicit feedback?
feature_columns = [
    "userId_index",
    "itemId_index",
    "age_index",
    "gender_index",
    "occupation_index",
]

feature_sizes = {
    "userId_index": len(ratings["userId_index"].unique()),
    "movieId_index": len(ratings["itemId_index"].unique()),
    "age_index": len(ratings["age_index"].unique()),
    "gender_index": len(ratings["gender_index"].unique()),
    "occupation_index": len(ratings["occupation_index"].unique()),
}

# * Set the second order FM model made of three parts:
# ? - 1. The offsets:
next_offset = 0
feature_offsets = {}

# * This is in order to establish when to pass to the next feature
for k, v in feature_sizes.items():
    feature_offsets[k] = next_offset
    next_offset += v

# * Map all column indices to start from correct offset
for column in feature_columns:
    ratings[column] = ratings[column].apply(lambda c: c + feature_offsets[column])

# * - Only visualise the feature columns along with the ratings, because that's what we need for FM.
ratings[[*feature_columns, "rating"]].head(5)

# * - Initialise the data and split it into train and test
data_x = torch.tensor(ratings[feature_columns].values)
data_y = torch.tensor(ratings["rating"].values).float()
dataset = data.TensorDataset(data_x, data_y)

bs = 1024
train_n = int(len(dataset) * 0.9)
valid_n = len(dataset) - train_n
splits = [train_n, valid_n]
assert sum(splits) == len(dataset)  # ? - Verify that the split has been done correctly
trainset, devset = torch.utils.data.random_split(
    dataset, splits
)  # ? - Assign the data to each split
train_dataloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
dev_dataloader = data.DataLoader(devset, batch_size=bs, shuffle=True)


# * Function to fill in a tensor with a 'truncated distribution' -> mean 0, std 1
# copied from fastai:
def trunc_normal_(x, mean=0.0, std=1.0):
    """
    Modifies a PyTorch tensor in-place, filling it with random values that approximate a truncated normal distribution.

    This function fills the tensor `x` with values drawn from a standard normal distribution, then applies a modulus operation to limit the absolute values, and finally scales and shifts these values to achieve the desired mean and standard deviation. Note that the approach does not strictly adhere to a statistically accurate truncated normal distribution, as it does not cut off values outside a specific range but rather wraps them within a limited range.

    Parameters:
    - x (Tensor): The PyTorch tensor to be modified in-place.
    - mean (float, optional): The mean of the distribution after adjustment. Defaults to 0.0.
    - std (float, optional): The standard deviation of the distribution after adjustment. Defaults to 1.0.

    Returns:
    - Tensor: The modified tensor `x` with values approximating a truncated normal distribution centered around `mean` and with a standard deviation of `std`. The tensor is modified in-place, so the return value is the same tensor object `x`.
    """
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class FMModel(nn.Module):
    def __init__(
        self, n, k
    ):  # ? - n: Number of unique features. k: Number of latent vectors
        super().__init__()

        self.w0 = nn.Parameter(torch.zeros(1))  # ? - Global bias
        self.bias = nn.Embedding(n, 1)  # ? - Embedding layer for bias per feature
        self.embeddings = nn.Embedding(
            n, k
        )  # ? - The actual embedding with dimension k

        # ? - This initialises the embeddings and bias layers with a truncated normal distribution
        with torch.no_grad():
            trunc_normal_(self.embeddings.weight, std=0.01)
        with torch.no_grad():
            trunc_normal_(self.bias.weight, std=0.01)

    def forward(
        self, X
    ):  # ? - How is the input tensor processed to produce a prediction?
        emb = self.embeddings(X)  # ? - Compute embeddings for the input features
        # ? - emb has shape: [batch_size, num_of_features, k]
        # calculate the interactions in complexity of O(nk) see lemma 3.1 from paper
        pow_of_sum = emb.sum(dim=1).pow(2)
        sum_of_pow = emb.pow(2).sum(dim=1)
        pairwise = (pow_of_sum - sum_of_pow).sum(1) * 0.5
        bias = self.bias(X).squeeze().sum(1)
        # I wrap the result with a sigmoid function to limit to be between 0 and 5.5.
        return torch.sigmoid(self.w0 + bias + pairwise) * 5.5

    # ? ^^Returns a sigmoid as the output will be limited between 0 and 1 -> The 5.5 I'm not sure why
    # ? Probably because of the rating prediction.


# fit/test functions
def fit(iterator, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for x, y in iterator:
        optimizer.zero_grad()
        y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item() * x.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss / len(iterator.dataset)


def test(iterator, model, criterion):
    train_loss = 0
    model.eval()
    for x, y in iterator:
        with torch.no_grad():
            y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item() * x.shape[0]
    return train_loss / len(iterator.dataset)


def train_n_epochs(model, n, optimizer, scheduler):
    criterion = nn.MSELoss().to(device)
    for epoch in range(n):
        start_time = time.time()
        train_loss = fit(train_dataloader, model, optimizer, criterion)
        valid_loss = test(dev_dataloader, model, criterion)
        scheduler.step()
        secs = int(time.time() - start_time)
        print(f"epoch {epoch}. time: {secs}[s]")
        print(f"\ttrain rmse: {(math.sqrt(train_loss)):.4f}")
        print(f"\tvalidation rmse: {(math.sqrt(valid_loss)):.4f}")


model = FMModel(data_x.max() + 1, 20).to(device)
wd = 1e-5
lr = 0.001
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)
criterion = nn.MSELoss().to(device)
for epoch in range(epochs):
    start_time = time.time()
    train_loss = fit(train_dataloader, model, optimizer, criterion)
    valid_loss = test(dev_dataloader, model, criterion)
    scheduler.step()
    secs = int(time.time() - start_time)
    print(f"epoch {epoch}. time: {secs}[s]")
    print(f"\ttrain rmse: {(math.sqrt(train_loss)):.4f}")
    print(f"\tvalidation rmse: {(math.sqrt(valid_loss)):.4f}")

# TODO: Aprender como ejecutar para coger la recomendación. Too abstract
