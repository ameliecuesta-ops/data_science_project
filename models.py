# models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def run_clustering(data, n_clusters=2):
    """Effectue un K-Means sur les données de consommation."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # On suppose que 'data' est un tableau de courbes
    clusters = model.fit_predict(data)
    return clusters, model

def train_forecasting(X, y):
    """Entraîne une régression linéaire pour la prédiction."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def generate_fake_consumption(type_residence):
    """Génère une courbe fictive selon le type (RP ou RS)."""
    heures = np.linspace(0, 23, 48) # 30 min steps
    if type_residence == "RP":
        # Profil avec pics matin et soir
        conso = 1 + np.sin(heures/3.5)**2 + np.random.normal(0, 0.1, 48)
    else:
        # Profil plat (Résidence Secondaire vide)
        conso = 0.2 + np.random.normal(0, 0.05, 48)
    return heures, conso