import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from models import run_clustering, train_forecasting, calculer_matrice_confusion, generate_fake_consumption

# Configuration de la page
st.set_page_config(page_title="Dashboard Enedis", layout="wide")

st.title("⚡ Projet Data Science : Analyse de Consommation")
st.sidebar.header("Options")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # Charge le fichier récupéré par Amélie
    df = pd.read_csv("RES2-6-9-labels.csv")
    return df

try:
    df = load_data()
    st.sidebar.success("Données chargées avec succès !")
except Exception as e:
    st.sidebar.error(f"Erreur de chargement : {e}")
    df = None

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["📊 Clustering & Evaluation", "📈 Prévision", "🪄 Générateur"])

if df is not None:
    # On suppose que la colonne de label est la dernière ou s'appelle 'label' / 'target'
    # On essaie de l'identifier automatiquement
    col_label = 'label' if 'label' in df.columns else df.columns[-1]
    X_data = df.drop(columns=[col_label])
    y_reel = df[col_label]

    with tab1:
        st.header("Clustering (Détection RS/RP)")
        st.write("Aperçu des données brutes :")
        st.dataframe(df.head(10))

        if st.button("Lancer l'algorithme K-Means"):
            # 1. Exécuter le clustering
            labels_predits = run_clustering(X_data)
            
            # 2. Afficher la matrice de confusion
            st.subheader("Matrice de Confusion")
            cm = calculer_matrice_confusion(y_reel, labels_predits)
            
            fig_cm = px.imshow(cm, 
                               text_auto=True, 
                               labels=dict(x="Prédiction (Cluster)", y="Réalité (Label)"),
                               x=['Classe 0', 'Classe 1'],
                               y=['Classe 0', 'Classe 1'],
                               color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_cm)
            st.info("Note : Les clusters 0 et 1 peuvent être inversés par rapport aux labels originaux.")

    with tab2:
        st.header("Prédiction de consommation")
        st.write("Modèle de Régression Linéaire basé sur une courbe moyenne.")
        
        # Exemple : on prend la moyenne de toutes les lignes pour s'entraîner
        conso_moyenne = X_data.mean().values
        heures = np.arange(len(conso_moyenne)).reshape(-1, 1)
        
        model_regr = train_forecasting(heures, conso_moyenne)
        
        h_pred = st.slider("Heure à prédire (pas de 30min)", 0, len(heures)-1, 10)
        res = model_regr.predict([[h_pred]])
        st.metric("Consommation estimée", f"{res[0]:.3f} kW")

    with tab3:
        st.header("Générateur de données synthétiques")
        choix = st.selectbox("Type de logement à simuler", ["RP", "RS"])
        
        if st.button("Générer une nouvelle courbe"):
            x_gen, y_gen = generate_fake_consumption(choix)
            fig_gen = px.line(x=x_gen, y=y_gen, title=f"Profil de consommation simulé : {choix}")
            fig_gen.update_layout(xaxis_title="Heures", yaxis_title="Puissance (kW)")
            st.plotly_chart(fig_gen)

else:
    st.warning("Veuillez vérifier que le fichier 'RES2-6-9-labels.csv' est bien dans le dossier.")