import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense  # type: ignore
from keras.optimizers import Adam  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    return df

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_sequences(X, max_length=150):
    return np.array([padded_one_hot_encode(seq.zfill(max_length)) for seq in X])

def padded_one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '0': [0, 0, 0, 0]}
    return np.array([mapping[nucleotide.upper()] for nucleotide in sequence])

def reshape_model_input(X):
    return np.array([[x, x, x, x] for x in X.values]).reshape(-1, 1, 4)

def concatenate_inputs(array1, array2):
    return np.concatenate((array1, array2), axis=1)

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
    return history

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss

def calc_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def plot_kde(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test.flatten(), label='y_test', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(y_pred.flatten(), label='y_pred', color='orange', fill=True, alpha=0.5)
    plt.xlabel('Observed log(TX/Txref)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_scatter(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='y = x line')
    plt.xlabel('Observed log(TX/Txref)')
    plt.ylabel('Predicted log(TX/Txref)')
    plt.title('Scatter plot of y_test vs y_pred')
    plt.grid()
    plt.show()

def plot_hexbin(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.hexbin(y_test.flatten(), y_pred.flatten(), gridsize=50, cmap='Blues', mincnt=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='y = x line')
    plt.colorbar(label='Counts')
    plt.xlabel('Observed log(TX/Txref)')
    plt.ylabel('Predicted log(TX/Txref)')
    plt.title('Hexbin plot of y_test vs y_pred')
    plt.grid()
    plt.show()