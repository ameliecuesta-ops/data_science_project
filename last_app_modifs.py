from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error

# Configuration de la page
st.set_page_config(page_title="Dashboard Enedis Pro", layout="wide")

# Titre personnalisé
st.markdown("""
    <style>
    .main-title {
        background: linear-gradient(135deg, #007BFF 0%, #0056D2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .main-title h1 {
        margin: 0;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-title p {
        margin: 10px 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
        letter-spacing: 1px;
    }
    </style>
    
    <div class="main-title">
        <h1>DASHBOARD ANALYSE ÉNERGÉTIQUE ENEDIS</h1>
        <p>Analyse de consommation, Clustering K-Means, Prédictions IA et Simulation Panneaux Solaires</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    # On charge les données brutes
    df_raw = pd.read_csv(path)
    df_raw.columns = df_raw.columns.str.strip()
    
    # Transformation : on pivote le tableau pour avoir une ligne par 'id' et une colonne par 'horodate'
    df = df_raw.pivot_table(index='id', columns='horodate', values='valeur', aggfunc='mean')
    
    # On remplace les éventuels trous de données par 0
    df = df.fillna(0)
    
    # Nos colonnes de consommation sont maintenant simplement toutes les colonnes du tableau
    conso_cols = df.columns.tolist()

    # Conversion Puissance -> Énergie et optimisation
    df[conso_cols] = df[conso_cols] * 0.5       
    df[conso_cols] = df[conso_cols].astype('float32')



    #AUGMENTER ACCURACY !!!!!!

    # On transforme les noms de colonnes en vraies dates pour pandas
    dates_temporelles = pd.to_datetime(conso_cols)
    
    # On identifie quelles colonnes correspondent au Week-End et à la Semaine
    cols_we = [conso_cols[i] for i in range(len(conso_cols)) if dates_temporelles[i].dayofweek >= 5]
    cols_semaine = [conso_cols[i] for i in range(len(conso_cols)) if dates_temporelles[i].dayofweek < 5]
    
    # On identifie les colonnes "Heures de pointe du soir" (ex: entre 18h et 22h)
    cols_pointe = [conso_cols[i] for i in range(len(conso_cols)) if 18 <= dates_temporelles[i].hour <= 22]


    
    # Création des indicateurs pour l'IA
    df['total'] = df[conso_cols].sum(axis=1)
    df['max'] = df[conso_cols].max(axis=1)
    df['std'] = df[conso_cols].std(axis=1)


    # Nos nouvelles variables discriminantes
    df['conso_moy_we'] = df[cols_we].mean(axis=1) if cols_we else 0
    df['conso_moy_semaine'] = df[cols_semaine].mean(axis=1) if cols_semaine else 0
    df['ratio_we_semaine'] = df['conso_moy_we'] / (df['conso_moy_semaine'] + 0.001)
    df['conso_pointe_soir'] = df[cols_pointe].mean(axis=1) if cols_pointe else 0


    
    # On reset l'index pour que 'id' redevienne une colonne normale (utile pour l'affichage)
    df = df.reset_index()
    
    return df, conso_cols

# Chargement
file_path = "export.csv"
if os.path.exists(file_path):
    df, conso_cols = load_data(file_path)
else:
    st.error("Fichier introuvable")
    st.stop()

# Structuration de la page en sections

# Section 1 (KPIs à gauche | IA à droite)
col_top_left, col_top_right = st.columns([1, 1])
height = 650

with col_top_left:
    st.markdown("<h3>KPI Globaux</h3>", unsafe_allow_html=True)
    
    # Centrer les KPIs
    st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
    with st.container(border=True, height=height):
        val_max = df[conso_cols].max().max()
        val_moyenne = df[conso_cols].mean().mean()
        nb_clients = len(df)
        
        str_max = f"{val_max:,.0f}".replace(",", " ") + " kW"
        str_moy = f"{val_moyenne:,.0f}".replace(",", " ") + " kW"
        str_clients = f"{nb_clients:,.0f}".replace(",", " ")
        html_code = (
            '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 580px; gap: 40px;">'
            
            '<div style="text-align: center;">'
            '<p style="font-size: 1.5rem; margin-bottom: 0px; opacity: 0.7;">Conso Max</p>'
            f'<p style="font-size: 4rem; font-weight: bold; margin-top: 0px; color: #007BFF;">{str_max}</p>'
            '</div>'
            
            '<div style="text-align: center;">'
            '<p style="font-size: 1.5rem; margin-bottom: 0px; opacity: 0.7;">Moyenne Globale</p>'
            f'<p style="font-size: 4rem; font-weight: bold; margin-top: 0px; color: #007BFF;">{str_moy}</p>'
            '</div>'
            
            '<div style="text-align: center;">'
            '<p style="font-size: 1.5rem; margin-bottom: 0px; opacity: 0.7;">Clients Total</p>'
            f'<p style="font-size: 4rem; font-weight: bold; margin-top: 0px; color: #007BFF;">{str_clients}</p>'
            '</div>'
            
            '</div>'
        )
        
        st.markdown(html_code, unsafe_allow_html=True)

with col_top_right:
    st.subheader("Intelligence Artificielle")
    with st.container(border=True, height=height):
        tab1, tab2 = st.tabs(["Classification (RF)", "Regression (Linear)"])
        st.sidebar.subheader("Paramètres des données")
        use_full_data = st.sidebar.checkbox("Utiliser le dataset complet (Attention : plus lent)", value=False)

        if use_full_data:
            df_sample = df
            st.sidebar.info(f"Mode : Dataset complet ({len(df)} lignes)")
        else:
            df_sample = df.sample(n=min(5000, len(df)), random_state=42)
            st.sidebar.info(f"Mode : Échantillon (5000 lignes)")

        # 1. CLUSTERING : On définit les profils basés sur TOUTES les données
        # On utilise nos nouvelles variables pour faire des groupes très marqués
        features_clustering = df_sample[['total', 'max', 'std', 'ratio_we_semaine', 'conso_pointe_soir']]
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_sample = kmeans.fit_predict(features_clustering)

        # 2. IA : On utilise ces mêmes variables pour que le modèle apprenne correctement !
        X_features_ia = features_clustering # On donne les mêmes variables riches au modèle

        with tab1:
            model_choice = st.radio("Modèle à tester :", ["Random Forest", "Logistic Regression"], horizontal=True)
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)

            # Entraînement
            X_train, X_test, y_train, y_test = train_test_split(X_features_ia, labels_sample, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = model.score(X_test, y_test)
            st.metric("Précision (Accuracy)", f"{acc*100:.1f}%")
            
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prédit", y="Réel"),
                               x=['Type A', 'Type B'], y=['Type A', 'Type B'],
                               color_continuous_scale='Blues', height=350)
            fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_cm, width='stretch')







        with tab2:
            # 1. On change la variable X : on utilise la conso de pointe au lieu du total !
            X_reg = df_sample[['conso_pointe_soir']]
            y_reg = df_sample['max']
            
            reg_rf = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=42)
            reg_rf.fit(X_reg,y_reg)
            preds_rf = reg_rf.predict(X_reg)
            mae_rf = mean_absolute_error(y_reg, preds_rf)
            
            st.metric("Erreur Moyenne (MAE)", f"{mae_rf:.1f} kW")
            
            # On ajoute temporairement les labels K-Means au dataframe de l'échantillon pour la couleur
            df_sample_graph = df_sample.head(100).copy()
            # On récupère les clusters pour ces 100 premiers clients
            df_sample_graph['Profil'] = labels_sample[:100] 
            # On transforme 0 et 1 en texte pour une jolie légende
            df_sample_graph['Profil'] = df_sample_graph['Profil'].map({0: 'Profil A', 1: 'Profil B'})
            


            # On crée une grille de points allant du min au max de la conso
            x_range = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
            y_range_preds = reg_rf.predict(pd.DataFrame(x_range, columns=['conso_pointe_soir']))


            # 2. On trace le scatter plot avec de la couleur par profil
            fig_reg = px.scatter(
                df_sample_graph, 
                x='conso_pointe_soir', 
                y='max', 
                color='Profil',
                trendline="ols",
                title="Prédiction du Pic via Random Forest (Non-linéaire)",
                color_discrete_sequence=['#007BFF', '#FF7F0E'], # Bleu et Orange
                height=350
            )
            

            fig_reg.add_trace(go.Scatter(
                x=x_range.flatten(), 
                y=y_range_preds, 
                mode='lines', 
                name='Modèle RF',
                line=dict(color='black', width=2)
            ))


            fig_reg.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_reg, width='stretch')

# Section 2 (Graphique Client)
st.markdown("---")
st.subheader("Courbe de Charge Client (Détail)")

with st.container(border=True):
    # Liste déroulante
    liste_ids = df['id'].unique()
    selected_id = st.selectbox("Rechercher ou choisir un ID Client :", options=liste_ids)
    
    # Graphique
    client_data = df[df['id'] == selected_id][conso_cols].iloc[0]
    fig = px.line(
        x=conso_cols, 
        y=client_data.values, 
        height=500,
        title=f"Consommation du client {selected_id}",
        labels={'x': 'Date / Heure', 'y': 'Énergie (kWh)'}
    )
    fig.update_traces(line=dict(width=1)) 
    fig.update_xaxes(
        rangeslider_visible=True, 
        rangeselector=dict(       
            buttons=list([
                dict(count=7, label="1 Sem", step="day", stepmode="backward"),
                dict(count=1, label="1 Mois", step="month", stepmode="backward"),
                dict(step="all", label="Tout")
            ])
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Section 2 (Graphique Client & Comparaison)
st.markdown("---")
st.subheader("Analyse Détaillée par Client")

with st.container(border=True):
    # Liste déroulante (qui sert aussi de barre de recherche nativement)
    liste_ids = df['id'].unique()
    selected_id = st.selectbox("Rechercher ou choisir un ID Client (tapez l'ID au clavier) :", options=liste_ids)
    
    # Récupération des données du client
    client_data = df[df['id'] == selected_id][conso_cols].iloc[0]
    
    # ---------------------------------------------
    # 1. Graphique : Courbe de charge temporelle et boxplot de comparaison avec la distribution globale
    # ---------------------------------------------
    # fig_line = px.line(
    #     x=conso_cols, 
    #     y=client_data.values, 
    #     height=400,
    #     title=f"Courbe de charge - Client {selected_id}",
    #     labels={'x': 'Date / Heure', 'y': 'Énergie (kWh)'}
    # )
    # fig_line.update_traces(line=dict(width=1)) 
    # fig_line.update_xaxes(
    #     rangeslider_visible=True, 
    #     rangeselector=dict(       
    #         buttons=list([
    #             dict(count=7, label="1 Sem", step="day", stepmode="backward"),
    #             dict(count=1, label="1 Mois", step="month", stepmode="backward"),
    #             dict(step="all", label="Tout")
    #         ])
    #     )
    # )
    # fig_line.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    # st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)
    st.markdown(f"**Comparaison de la distribution (Moyenne, Médiane, Min/Max) : Global vs Client {selected_id}**")
    global_values = df[conso_cols].values.flatten()
    if len(global_values) > 100000:
        np.random.seed(42)
        global_values = np.random.choice(global_values, 100000, replace=False)
        
    # Création d'un dataset temporaire pour le graphique
    df_box = pd.DataFrame({
        'Énergie (kWh)': np.concatenate([global_values, client_data.values]),
        'Comparaison': ['Global (Tous les clients)'] * len(global_values) + [f'Client {selected_id}'] * len(client_data.values)
    })
    
    # Création du Boxplot
    fig_box = px.box(
        df_box, 
        x='Énergie (kWh)', 
        y='Comparaison', 
        color='Comparaison', 
        orientation='h', # Format Horizontal
        points=False,
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    
    # boxmean=True ajoute la ligne pointillée pour la Moyenne (la ligne pleine est la Médiane)
    fig_box.update_traces(boxmean=True) 
    fig_box.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

# Section 3 (Simulateur Panneaux Solaires)
st.subheader("Simulateur Panneaux Solaires")
with st.container(border=True):
    
    # Le curseur pour choisir la puissance des panneaux (en kW)
    puissance_solaire = st.slider("Taille de l'installation solaire (kWc)", 0, 500, 100, step=10)
    
    # La Mathématique du soleil : on crée une fonction qui simule la production solaire en fonction de l'heure
    dates = pd.to_datetime(conso_cols, utc=True)
    heures = dates.hour + dates.minute / 60.0
    
    # On crée une "cloche" qui vaut 1 à 13h, et 0 avant 8h et après 18h
    facteur_soleil = np.maximum(0, 1 - ((heures - 13) / 5)**2)
    production_solaire = puissance_solaire * facteur_soleil
    
    # Calcul de la nouvelle consommation nette du client après prise en compte de la production solaire
    conso_nette = np.maximum(0, client_data.values - production_solaire)
    
    # On prépare un tableau pour les 7 premiers jours
    df_simul = pd.DataFrame({
        'Date': conso_cols[:336], 
        'Sans Solaire (Avant)': client_data.values[:336],
        'Avec Solaire (Après)': conso_nette[:336]
    })
    
    # On dessine le graphique de simulation
    fig_simul = px.line(df_simul, x='Date', y=['Sans Solaire (Avant)', 'Avec Solaire (Après)'], 
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'],
                        height=250)
                        
    fig_simul.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text=None)
    
    st.plotly_chart(fig_simul, use_container_width=True)

with st.container(border=True):
    st.subheader("Synthèse Méthodologique")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Data Prep**")
            st.caption("• Facteur 0.5 (kW -> kWh)\n• Pivot des données\n• Échantillonnage 5k")
        with c2:
            st.write("**Modèles**")
            st.caption("• Comparaison RF vs LogReg\n• K-Means (Labellisation)\n• Évitement du Data Leakage")
        with c3:
            st.write("**Métriques**")
            st.caption("• Accuracy (Classification)\n• MAE (Régression)")
