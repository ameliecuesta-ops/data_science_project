import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def run_clustering(data, n_clusters=2):
    """Effectue un K-Means sur les données."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(data)
    return clusters

def train_forecasting(X, y):
    """Entraîne une régression linéaire."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def calculer_matrice_confusion(y_reel, y_pred):
    """Calcule la matrice de confusion."""
    return confusion_matrix(y_reel, y_pred)

def generate_fake_consumption(type_residence):
    """Génère une courbe fictive (48 points pour 24h par pas de 30min)."""
    x = np.linspace(0, 23.5, 48)
    if type_residence == "RP":
        # Profil type : deux pics (matin et soir)
        y = 0.5 + 2*np.exp(-(x-8)**2/4) + 3*np.exp(-(x-20)**2/5) + np.random.normal(0, 0.1, 48)
    else:
        # Profil type : consommation basse et stable
        y = 0.2 + np.random.normal(0, 0.05, 48)
    return x, y