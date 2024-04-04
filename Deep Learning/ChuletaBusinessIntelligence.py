# ************************************
# ****** Chuleta Yago Tobio **********
# ************************************

# ? - Librerias
# Librerías necesarias
import pandas as pd
import numpy as np
import random
import sklearn

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import sklearn.metrics

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Embedding, Concatenate, Reshape
import keras.metrics

from tensorflow import keras
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ********************** INDEX *************************

# ******************************************************

# ! - Sequential model with Keras - Stack lineal
# ? - Util para capas donde cada una tiene un tensor de entrada y uno de salida.
# ? - Capas en uso: Flatten, Dense, Dropout

# ? - Logits: Output de la última capa de la red neuronal tras haber aplicado la func. de activación
# ! - Types of activation functions:
# * sigmoid -> Used for binary classification tasks (Not great. Slow convergence, saturated gradients)
# * tanh -> Fixes the fact that sigmoid is not zero-centerd, still suffers vanishing gradient.
# * ReLu -> The most common output layer. Helps overcome vanishing gradient.
# * softmax -> Used for multi-output classifications.
model = tf.keras.models.Sequential(
    [
        # ? - Con la función .Flatten, aplanamos la imagen 2D a un vector 1D de 784 elementos -> Ya que es sequential. Un tensor de entrada y uno de salida.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # ? - Esta capa esta completamente interconectada con 128 neuronas, aplicando la función RELU -> Relación no lineal.
        tf.keras.layers.Dense(128, activation="relu"),
        # ? - Esta capa añade suelta un 20% de las conexiones de la capa anterior para prevenir overfitting.
        tf.keras.layers.Dropout(0.2),
        # ? - La salida es una capa densa con las 10 salidas que deseamos. Vamos definiendo capa a capa.
        tf.keras.layers.Dense(10),
    ]
)

# * You can also do it like this:
model = tf.keras.models.Sequential(name="sequential_model")
model.add(
    tf.keras.layers.Dense(
        10, input_shape=(50,), activation="relu", name="hidden_layer_1"
    )
)
model.add(tf.keras.layers.Dense(5, activation="relu", name="hidden_layer_2"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="final_layer"))

# ! - Functional Model with Keras -> Complex network of pipes for multiple tensors
# ? - Define the input layer of the model. The shape=(50, ) parameter indicates that each input data instance is a 1-dim array of 50 features.
input_layer = tf.keras.layers.Input(shape=(50,), name="input")
# ? - Here we add dense layers to the model with 10 neurons. Outputs and inputs are re-fed into it.
# ? - As no activation function has been defined in each layer, Keras defaults to a linear activation function f(x) = x
x_1 = tf.keras.layers.Dense(10, name="hidden_layer_1")(input_layer)  # * Hidden Layer 1
x_2 = tf.keras.layers.Dense(10, name="hidden_layer_2")(x_1)  # * Hidden Layer 2
x_3 = tf.keras.layers.Dense(10, name="final_layer")(x_2)
# * Output Layer, however no density function has been indicated
model = tf.keras.models.Model(input_layer, x_3, name="functional_model")
model.summary()

# ! - Functional Model with Keras for various types of data:
input_numerical = tf.keras.layers.Input(shape=(1,), name="numerical_input")
input_categorical = tf.keras.layers.Input(shape=(1,), name="categorical_input")

# ? - For the numeric variables, we create a 10-neuron dense layer with the tanh activation function
x_numeric = tf.keras.layers.Dense(10, activation="tanh", name="encoding_numerical")(
    input_numerical
)

# ? - For the categorical we apply an embedding layer, and a reshape layer, explanations in the following block
x_categorical = tf.keras.layers.Embedding(
    input_dim=5, output_dim=3, name="embedding_categorical"
)(input_categorical)
x_categorical = tf.keras.layers.Reshape(target_shape=(3,), name="flat_vector")(
    x_categorical
)
# ? - We then merge thse layers, and apply 2 final dense layers for a single output.
x = tf.keras.layers.Concatenate()([x_numeric, x_categorical])
x = tf.keras.layers.Dense(10, activation="tanh")(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(
    [input_numerical, input_categorical], x, name="model_with_two_inputs"
)

model.summary()

# ? - Model Compilation + Loss Function
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),  # * This is the loss function -> You can also just write Binary CrossEntropy
    metrics=["accuracy", tf.keras.metrics.AUC(name="area_under_curve")],
)

# ? -  Model fit
# * 1. Probable use case for a sequential API model.
model.fit(X_train, y_train)
# * 2. Use case for a functional API model with 2 types of input and a single output
model.fit((X_train_numerical, X_train_categorical), y_train)
# * 3. This is the same as before, however we're now defining the input as a dictionary to specify it more in detail.
model.fit(
    {"numerical_input": X_train_numerical, "categorical_input": X_train_numerical},
    y_train,
)
# * 4. Introduction to the concept of a validation split => Proxy to the test dataset.
model.fit(X_train, y_train, validation_split=0.2)
model.fit(X_train, y_train, validation_data=(X_val, y_val))

model.evaluate(x_test, y_test, verbose=2)  # type: ignore


# * Model Evaluation
# ? - Lo del parametro verbose controla cuanto output se ve en la consola durante el proceso de evaluación.
# ? - Verbose = 0. No output
# ? - Verbose = 1, Barra de progreso
# ? - Verbose = 2, menciona el numero del epoch.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # type: ignore

print("\nTest accuracy:", test_acc)


# * Model Predictions:
# ? - Aquí hemos aplicado la función de activación softmax
# * -> You can see how we have passed from logit to probability thanks to SoftMax
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  # type: ignore
