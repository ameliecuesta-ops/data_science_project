# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
# On importe tes fonctions IA depuis l'autre fichier
from models import train_forecasting, generate_fake_consumption

st.set_page_config(page_title="Dashboard Enedis AI", layout="wide")

st.title("⚡ Analyse de Consommation Enedis")
st.sidebar.info("Projet réalisé par Amélie & Gaëtan")

# Création des onglets
tab1, tab2, tab3 = st.tabs(["📊 Clustering & Classif", "📈 Forecasting", "🪄 Génération"])

with tab1:
    st.header("Détection RS vs RP")
    st.write("Ici s'affichera le clustering une fois le dataset chargé.")
    # On pourra appeler run_clustering(df) ici

with tab2:
    st.header("Prédiction de la consommation")
    # Petit exemple avec ta régression linéaire
    st.write("Test de la régression linéaire sur 24h :")
    
    # Données fictives pour le test
    X = np.array(range(24)).reshape(-1, 1)
    y = np.array([0.5, 0.4, 0.3, 0.2, 0.5, 1.2, 2.0, 2.5, 2.1, 1.8, 1.5, 1.6, 1.4, 1.3, 1.2, 1.5, 2.2, 2.8, 3.2, 3.0, 2.5, 1.8, 1.2, 0.8])
    
    model_regr = train_forecasting(X, y)
    
    heure_test = st.slider("Prédire pour quelle heure ?", 0, 23, 12)
    pred = model_regr.predict([[heure_test]])
    st.metric(label=f"Consommation prévue à {heure_test}h", value=f"{pred[0]:.2f} kW")

with tab3:
    st.header("Générateur de courbes")
    type_choisi = st.radio("Sélectionnez un type de client :", ["RP", "RS"])
    
    if st.button("Générer une courbe"):
        h, c = generate_fake_consumption(type_choisi)
        df_gen = pd.DataFrame({"Heure": h, "Conso (kW)": c})
        fig = px.line(df_gen, x="Heure", y="Conso (kW)", title=f"Profil généré : {type_choisi}")
        st.plotly_chart(fig)