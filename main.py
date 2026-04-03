import streamlit as st
import os
import backend
import frontend

# 1. Configuration de la page
frontend.set_page_config_and_title()

# 2. Chargement de la donnée (C'est ici qu'on applique le cache pour Streamlit)
@st.cache_data
def load_and_cache_data(path):
    return backend.process_data(path)

dossier_actuel = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dossier_actuel, 'export.csv')

if os.path.exists(file_path):
    df, conso_cols = load_and_cache_data(file_path)
else:
    st.error(f"Fichier introuvable au chemin : {file_path}")
    st.stop()

# --- Section 1 (KPIs à gauche | IA à droite) ---
col_top_left, col_top_right = st.columns([1, 1])
height = 650

with col_top_left:
    st.markdown("<h3>KPI Globaux</h3>", unsafe_allow_html=True)
    with st.container(border=True, height=height):
        val_max, val_moyenne, nb_clients = backend.get_kpis(df, conso_cols)
        frontend.display_kpi_section(val_max, val_moyenne, nb_clients)

with col_top_right:
    st.subheader("Intelligence Artificielle")
    with st.container(border=True, height=height):
        tab1, tab2 = st.tabs(["Classification (RF)", "Regression (Linear)"])
        
        st.sidebar.subheader("Paramètres des données")
        use_full_data = st.sidebar.checkbox("Utiliser le dataset complet (Attention : plus lent)", value=False)
        
        df_sample = backend.prepare_sample(df, use_full_data)
        if use_full_data:
            st.sidebar.info(f"Mode : Dataset complet ({len(df)} lignes)")
        else:
            st.sidebar.info(f"Mode : Échantillon (5000 lignes)")

        with tab1:
            model_choice = st.radio("Modèle à tester :", ["Random Forest", "Logistic Regression"], horizontal=True)
            acc, cm, labels_sample = backend.train_classification(df_sample, model_choice)
            frontend.display_classification_tab(acc, cm)
            
        with tab2:
            mae_rf, x_range, y_range_preds, df_sample_graph = backend.train_regression(df_sample, labels_sample)
            frontend.display_regression_tab(mae_rf, x_range, y_range_preds, df_sample_graph)

# --- Section 2 (Graphique Client) ---
st.markdown("---")
st.subheader("Courbe de Charge Client (Détail)")
with st.container(border=True):
    liste_ids = df['id'].unique()
    selected_id_2 = st.selectbox("Rechercher ou choisir un ID Client :", options=liste_ids)
    client_data_2 = df[df['id'] == selected_id_2][conso_cols].iloc[0]
    frontend.display_client_curve(client_data_2, conso_cols, selected_id_2)

# --- Section 3 : classification du profil ---
st.markdown("---")
st.markdown("<h4 style='margin-top:0;'>Définition du profil selon la consommation d'éléctricité au rythme horaire</h4>", unsafe_allow_html=True)
with st.container(border=True):
    selected_id_3 = st.selectbox("Rechercher ou choisir un ID Client (tapez l'ID au clavier) :", options=liste_ids, key="select_section_3")
    client_data_3 = df[df['id'] == selected_id_3][conso_cols].iloc[0]
    type_client, explication, couleur_bord, conso_nuit, conso_journee, conso_soiree = backend.analyze_client_profile(client_data_3, conso_cols)
    frontend.display_client_profile_info(type_client, explication, couleur_bord, conso_nuit, conso_journee, conso_soiree)

# --- Section 4 : Analyse détaillée ---
st.markdown("---")
st.subheader("Analyse Détaillée par Client")
with st.container(border=True):
    selected_id_4 = st.selectbox("Rechercher ou choisir un ID Client (tapez l'ID au clavier) :", options=liste_ids, key="select_section_4")
    client_data_4 = df[df['id'] == selected_id_4][conso_cols].iloc[0]
    frontend.display_detailed_analysis(selected_id_4, client_data_4, df, conso_cols)

# --- Section 5 : Simulateur ---
st.subheader("Simulateur Panneaux Solaires")
with st.container(border=True):
    puissance_solaire = st.slider("Taille de l'installation solaire (kWc)", 0, 500, 100, step=10)
    df_simul = backend.simulate_solar_production(client_data_4, conso_cols, puissance_solaire)
    frontend.display_solar_simulation(df_simul, client_data_4)


# --- NOUVEAUX BLOCS

# 1. Bloc PCA (Juste après l'IA existante)
st.markdown("---")
st.subheader("Analyse de Dimensionnalité (PCA)")
df_pca = backend.run_pca(df_sample)
frontend.display_pca_chart(df_pca, labels_sample)

# 2. Bloc Forecasting (Juste après ton analyse détaillée client)
st.markdown("---")
st.subheader("Prédiction de consommation (Forecasting)")
y_r, y_p = backend.train_forecasting(client_data_4)
frontend.display_forecasting(y_r, y_p)

# --- Section 2b : Heatmap hebdomadaire ---
st.markdown("---")
st.subheader("🗓️ Heatmap de charge — heure × jour de la semaine")
with st.container(border=True):
    st.caption("Consommation moyenne (kWh) par créneau horaire, tous clients et toutes semaines confondus.")
    df_heatmap = backend.get_heatmap_data(df, conso_cols)
    frontend.display_load_heatmap(df_heatmap)

# 3. Bloc Générateur (Avant la synthèse finale)
st.markdown("---")
st.subheader("Générateur de profils types (RP/RS)")
type_gen = st.radio("Type à simuler", ["Résidence Principale (RP)", "Résidence Secondaire (RS)"], horizontal=True)
t_gen, v_gen = backend.generate_synthetic_profile(type_gen)
frontend.display_generator_section(t_gen, v_gen, type_gen)

# --- Section 6 ---
frontend.display_synthesis()
