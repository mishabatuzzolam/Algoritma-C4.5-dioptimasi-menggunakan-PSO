!pip install pyswarm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso

# Memuat data dari Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Memuat dataset
dataset_path = "/content/drive/My Drive/janin.csv"
dataset = pd.read_csv(dataset_path, sep=';')

# Asumsikan kolom terakhir adalah kolom target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Membagi data menjadi data uji dan data latih
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Tentukan fungsi fitness untuk optimasi PSO
def fitness_function(weights, X_train, X_test, y_train, y_test):
    selected_features = weights > 0.5  # Ambang batas untuk memilih fitur
    selected_columns = [i for i, selected in enumerate(selected_features) if selected]
    X_train_selected = X_train[:, selected_columns]
    X_test_selected = X_test[:, selected_columns]
    if np.sum(selected_features) == 0:
        return 1e6  # Kembalikan nilai tinggi jika tidak ada fitur yang dipilih
    else:
        clf = DecisionTreeClassifier()
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)
        return 1 - accuracy_score(y_test, y_pred)  # Minimalkan tingkat kesalahan klasifikasi

# Tentukan fungsi optimasi PSO
def pso_feature_selection(X_train, X_test, y_train, y_test, num_particles=10, max_iter=100):
    num_features = X_train.shape[1]
    lb = np.zeros(num_features)  # Batas bawah untuk fitur (0 untuk tidak dipilih)
    ub = np.ones(num_features)   # Batas atas untuk fitur (1 untuk yang dipilih)
    # Define objective function for PSO
    objective_function = lambda weights: fitness_function(weights, X_train, X_test, y_train, y_test)
    # Perform PSO optimization
    weights_opt, _ = pso(objective_function, lb, ub, maxiter=max_iter, swarmsize=num_particles)
    return weights_opt

# Lakukan pemilihan fitur menggunakan PSO
selected_weights = pso_feature_selection(X_train, X_test, y_train, y_test)

selected_features = selected_weights > 0.5
selected_columns = [i for i, selected in enumerate(selected_features) if selected]
X_train_selected = X_train[:, selected_columns]
X_test_selected = X_test[:, selected_columns]

# Latih pengklasifikasi pohon keputusan dengan fitur yang dipilih
clf = DecisionTreeClassifier()
clf.fit(X_train_selected, y_train)

# Evaluasi model
y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
