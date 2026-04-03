import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def process_data(path):
    # On charge les données brutes
    df_raw = pd.read_csv(path)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Transformation : on pivote le tableau
    df = df_raw.pivot_table(index='id', columns='horodate', values='valeur', aggfunc='mean')
    df = df.fillna(0)
    conso_cols = df.columns.tolist()

    # Conversion Puissance -> Énergie et optimisation
    df[conso_cols] = df[conso_cols] * 0.5       
    df[conso_cols] = df[conso_cols].astype('float32')

    # On transforme les noms de colonnes en vraies dates pour pandas
    dates_temporelles = pd.to_datetime(conso_cols)
    
    cols_we = [conso_cols[i] for i in range(len(conso_cols)) if dates_temporelles[i].dayofweek >= 5]
    cols_semaine = [conso_cols[i] for i in range(len(conso_cols)) if dates_temporelles[i].dayofweek < 5]
    cols_pointe = [conso_cols[i] for i in range(len(conso_cols)) if 18 <= dates_temporelles[i].hour <= 22]

    # Création des indicateurs pour l'IA
    df['total'] = df[conso_cols].sum(axis=1)
    df['max'] = df[conso_cols].max(axis=1)
    df['std'] = df[conso_cols].std(axis=1)

    # Variables discriminantes
    df['conso_moy_we'] = df[cols_we].mean(axis=1) if cols_we else 0
    df['conso_moy_semaine'] = df[cols_semaine].mean(axis=1) if cols_semaine else 0
    df['ratio_we_semaine'] = df['conso_moy_we'] / (df['conso_moy_semaine'] + 0.001)
    df['conso_pointe_soir'] = df[cols_pointe].mean(axis=1) if cols_pointe else 0

    df = df.reset_index()
    return df, conso_cols

def get_kpis(df, conso_cols):
    val_max = df[conso_cols].max().max()
    val_moyenne = df[conso_cols].mean().mean()
    nb_clients = len(df)
    return val_max, val_moyenne, nb_clients

def prepare_sample(df, use_full_data):
    if use_full_data:
        return df
    else:
        return df.sample(n=min(5000, len(df)), random_state=42)

def train_classification(df_sample, model_choice):
    features_clustering = df_sample[['total', 'max', 'std', 'ratio_we_semaine', 'conso_pointe_soir']]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_sample = kmeans.fit_predict(features_clustering)
    
    X_features_ia = features_clustering 

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    X_train, X_test, y_train, y_test = train_test_split(X_features_ia, labels_sample, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm, labels_sample

def train_regression(df_sample, labels_sample):
    X_reg = df_sample[['conso_pointe_soir']]
    y_reg = df_sample['max']
    
    reg_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg_rf.fit(X_reg, y_reg)
    preds_rf = reg_rf.predict(X_reg)
    mae_rf = mean_absolute_error(y_reg, preds_rf)
    
    x_range = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
    y_range_preds = reg_rf.predict(pd.DataFrame(x_range, columns=['conso_pointe_soir']))
    
    df_sample_graph = df_sample.head(100).copy()
    df_sample_graph['Profil'] = labels_sample[:100] 
    df_sample_graph['Profil'] = df_sample_graph['Profil'].map({0: 'Profil A', 1: 'Profil B'})
    
    return mae_rf, x_range, y_range_preds, df_sample_graph

def analyze_client_profile(client_data, conso_cols):
    conso_valeurs = client_data.values
    heures_extraites = [d.hour for d in pd.to_datetime(conso_cols)]
    
    df_client_horaire = pd.DataFrame({'Heure': heures_extraites, 'Conso': conso_valeurs})
    journee_type_client = df_client_horaire.groupby('Heure')['Conso'].mean()
    
    conso_nuit = journee_type_client.loc[0:5].mean() 
    conso_journee = journee_type_client.loc[8:17].mean() 
    conso_soiree = journee_type_client.loc[18:22].mean() 
    
    talon_conso = journee_type_client.min()
    seuil_pic = talon_conso + (journee_type_client.max() - talon_conso) * 0.5
    heures_de_pics = journee_type_client[journee_type_client > seuil_pic].index.tolist()
    
    type_client = "Commerce de proximité"
    explication = "La consommation présente des pics irréguliers au cours de la journée. Typique d'un commerce qui s'adapte à ses clients."
    couleur_bord = "#FFC107"

    if journee_type_client.max() < (talon_conso * 1.5):
        type_client = "Site Industriel / Activité continue"
        explication = "La courbe est très plate et lissée. Il n'y a pas de gros pics qui se détachent, l'activité tourne en 24/7."
        couleur_bord = "#DC3545"
    elif heures_de_pics:
        pics_soir = sum(1 for h in heures_de_pics if h >= 18 or h <= 2)
        pics_journee = sum(1 for h in heures_de_pics if 8 <= h <= 17)
        
        if pics_soir > (pics_journee * 1.5) and pics_soir > 0:
            type_client = "Profil Résidentiel (Foyers)"
            explication = "Les pics de consommation les plus violents ont lieu en soirée. C'est le comportement classique de particuliers qui rentrent chez eux."
            couleur_bord = "#28A745"
        elif pics_journee > (pics_soir * 1.2) and pics_journee > 0:
            type_client = "Entreprise / Bureaux"
            explication = "Les pics de consommation sont concentrés pendant les heures de bureau. Signature typique d'une activité pro de jour."
            couleur_bord = "#007BFF"
        else:
            type_client = "Commerce de proximité"
            explication = "La consommation est répartie avec des pics à la fois en journée et en soirée. Typique d'un commerce de quartier."
            couleur_bord = "#FFC107"
            
    return type_client, explication, couleur_bord, conso_nuit, conso_journee, conso_soiree

def simulate_solar_production(client_data, conso_cols, puissance_solaire):
    dates_completes = pd.to_datetime(conso_cols, utc=True)
    heures_completes = dates_completes.hour + dates_completes.minute / 60.0
    
    facteur_soleil = np.maximum(0, 1 - ((heures_completes - 13) / 5)**2)
    production_solaire = puissance_solaire * facteur_soleil
    conso_nette = np.maximum(0, client_data.values - production_solaire)
    
    df_simul = pd.DataFrame({
        'Date': conso_cols, 
        'Sans Solaire (Avant)': client_data.values,
        'Avec Solaire (Après)': conso_nette
    })
    return df_simul

def run_pca(df_sample):
    features = df_sample[['total', 'max', 'std', 'ratio_we_semaine', 'conso_pointe_soir']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features_scaled)
    return pd.DataFrame(pca_results, columns=['PC1', 'PC2'])

def train_forecasting(client_data):
    ts = client_data.values
    X = ts[:-1].reshape(-1, 1)
    y = ts[1:]
    model = LinearRegression()
    model.fit(X, y)
    return y, model.predict(X)

def generate_synthetic_profile(type_client):
    t = np.linspace(0, 24, 48)
    base = np.random.normal(0.5, 0.1, 48)
    if "Principale" in type_client:
        base += 1.5 * np.exp(-((t-8)**2)/2) + 2.5 * np.exp(-((t-20)**2)/2)
    else:
        base *= 0.4
    return t, base
