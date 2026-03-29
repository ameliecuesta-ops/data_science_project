from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION -------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Enedis Pro", layout="wide")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    conso_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # On multiplie par 0.5 car 30 min = 0.5 heure (Conversion Puissance -> Énergie)
    df[conso_cols] = df[conso_cols] * 0.5	    
    
    # On force le type float32 pour économiser la RAM
    df[conso_cols] = df[conso_cols].astype('float32')
    
    # On crée les indicateurs pour l'IA
    df['total'] = df[conso_cols].sum(axis=1)
    df['max'] = df[conso_cols].max(axis=1)
    df['std'] = df[conso_cols].std(axis=1)
    
    return df, conso_cols





# Chargement
file_path = "export.csv"
if os.path.exists(file_path):
    df, conso_cols = load_data(file_path)
else:
    st.error("Fichier introuvable")
    st.stop()

#--------------------------------------------------------------------------------------------




# --- STRUCTURE DU DASHBOARD (LE LAYOUT) ---------------------------------------------------

# 1. LIGNE DU HAUT (Classification/Regressor à gauche | Graphics à droite)
col_top_left, col_top_right = st.columns([1, 1])


with col_top_left:
    st.subheader("🤖 Intelligence Artificielle")
    with st.container(border=True):
        tab1, tab2 = st.tabs(["Classification (RF)", "Regression (Linear)"])
        
        # --- PREPARATION DES LABELS VIA CLUSTERING (JUSTIFICATION MÉTHODO) ---
        # On crée des labels 'Type A' et 'Type B' basés sur le comportement
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import confusion_matrix, mean_absolute_error


        # Utilisation d'un échantillon pour la fluidité (df et non sf !)
        df_sample = df.sample(5000) if len(df) > 5000 else df
        features_sample = df_sample[['total', 'max', 'std']]
        
        # Création des labels via Clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_sample = kmeans.fit_predict(features_sample)



        with tab1:
            
            # 1. CHOIX DU MODÈLE (Exigence Prof : Comparer plusieurs modèles)
            st.markdown("**Sélection du Modèle**")
            model_choice = st.radio("Modèle à tester :", ["Random Forest", "Logistic Regression"], horizontal=True)
            
            # Justification dynamique selon le choix
            if model_choice == "Random Forest":
                st.caption("Justification : Robuste aux valeurs aberrantes et capture les relations non-linéaires.")
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                st.caption("Justification : Modèle linéaire simple, idéal pour établir une base de référence (baseline).")
                from sklearn.linear_model import LogisticRegression # Import local pour être sûr
                model = LogisticRegression(max_iter=1000)

            # 2. ENTRAÎNEMENT & PRÉDICTION
            X_train, X_test, y_train, y_test = train_test_split(features_sample, labels_sample, test_size=0.2, random_state=42)
            
            # On entraîne le modèle choisi
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 2. MÉTRIQUES D'ÉVALUATION
            acc = model.score(X_test, y_test)
            st.metric("Précision (Accuracy)", f"{acc*100:.1f}%")
            
            # Matrice de confusion (Exigence Prof)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prédit", y="Réel"),
                               x=['Type A', 'Type B'], y=['Type A', 'Type B'],
                               color_continuous_scale='Blues', height=200)
            fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cm, width='stretch')






        with tab2:
            # 1. MODÈLE & JUSTIFICATION
            st.markdown("**Modèle : Régression Linéaire**")
            st.caption("Objectif : Prédire la puissance Max en fonction de la conso Totale.")
            
            # Regression : prédire 'max' à partir de 'total'
            reg = LinearRegression().fit(features_sample[['total']], features_sample['max'])
            preds = reg.predict(features_sample[['total']])
            mae = mean_absolute_error(features_sample['max'], preds)
            
            # 2. MÉTRIQUES D'ÉVALUATION
            st.metric("Erreur Moyenne (MAE)", f"{mae:.3f} kW")
            
            # Graphique de régression
            fig_reg = px.scatter(df_sample.head(100), x='total', y='max', trendline="ols",
                                 title="Prédiction du Pic de Charge (Max) via Conso Totale", height=200)
            st.plotly_chart(fig_reg, width='stretch')



with col_top_right:
    st.subheader("📈 Graphics")
    with st.container(border=True):
        client_id = st.number_input("ID Client", 0, len(df)-1, 0)
        fig = px.line(df.iloc[client_id][conso_cols], height=300)
        st.plotly_chart(fig, width='stretch')





# 2. LIGNE DU MILIEU / BAS (Generator à gauche | KPI Metrics à droite)
col_bot_left, col_bot_right = st.columns([2, 1])

with col_bot_left:
    st.subheader("🪄 Bloc Generator")
    with st.container(border=True):
        noise = st.slider("Variation", 0.0, 1.0, 0.2)
        gen_data = df[conso_cols].mean() + np.random.normal(0, noise, len(conso_cols))
        st.line_chart(gen_data, height=200)

with col_bot_right:
    st.subheader("📊 KPI Metrics")
    with st.container(border=True):
        st.metric("Conso Max", f"{df[conso_cols].max().max():.2f} kW")
        st.metric("Moyenne Globale", f"{df[conso_cols].mean().mean():.2f} kW")
        st.metric("Clients Total", len(df))



# 3. BLOC RECTANGLE EN BAS (Largeur complète)
st.subheader("📋 Synthèse Méthodologique & Justifications")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Data Prep**")
        st.caption("Facteur 0.5 appliqué (kW -> kWh).")
        st.caption("Échantillonnage 5k pour fluidité.")
    with c2:
        st.write("**Modèles**")
        st.caption("Comparaison RF vs LogReg.")
        st.caption("K-Means pour labellisation RS/RP.")
    with c3:
        st.write("**Métriques**")
        st.caption("Accuracy pour classification.")
        st.caption("MAE pour le forecasting.")




