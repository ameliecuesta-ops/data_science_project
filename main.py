"""
main.py — Point d'entree principal du dashboard Enedis.
Orchestre le chargement des donnees (backend) et le rendu (frontend).
"""

import os
import streamlit as st
import numpy as np

import backend
import frontend

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
frontend.set_page_config_and_title()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
page, load_all = frontend.render_sidebar(backend.N_INITIAL, backend.N_SAMPLE)

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNEES (cache Streamlit)
# ══════════════════════════════════════════════════════════════════════════════
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export.csv')

if not os.path.exists(FILE_PATH):
    st.error(f"Fichier introuvable : `{FILE_PATH}`. Placez `export.csv` a la racine du projet.")
    st.stop()


@st.cache_resource(show_spinner=False)
def _load(path, n):
    return backend.load_houses(path, n=n)


@st.cache_resource(show_spinner=False)
def _sample(path, n, sample_n, seed=42):
    df_all, list_ids = _load(path, n)
    df_work, work_ids = backend.get_work_sample(df_all, list_ids, sample_n, seed)
    return df_all, list_ids, df_work, work_ids


_n = None if load_all else backend.N_INITIAL
df_all, list_ids, df_work, work_ids = _sample(FILE_PATH, _n, backend.N_SAMPLE)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Exploration":
    frontend.display_exploration_header()

    kpis = backend.get_exploration_kpis(df_all, list_ids, work_ids)
    frontend.display_exploration_kpis(kpis)

    if not load_all:
        st.caption(
            f"Affichage des **{len(list_ids)}** premiers compteurs. "
            f"Utilisez **Charger tous les compteurs** dans la barre laterale pour inclure le dataset complet. "
            f"Les pages ML utilisent un echantillon reproductible de **{len(work_ids)}** compteurs."
        )

    st.divider()

    # Apercu CSV
    with st.container(border=True):
        st.subheader("Apercu du fichier brut")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("5 lignes aleatoires"):
                st.session_state.skip_rows = int(np.random.randint(1, 50_000))
        skip   = st.session_state.get('skip_rows', 0)
        df_prev = backend.load_preview(FILE_PATH, skip=skip)
        frontend.display_raw_preview(df_prev)

    # Distribution globale
    frontend.display_power_distribution(df_all)

    # Signature annuelle d'un compteur
    with st.container(border=True):
        st.subheader("Signature annuelle d'un compteur")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("Autre compteur"):
                st.session_state.selected_id = np.random.choice(list_ids)
        if 'selected_id' not in st.session_state:
            st.session_state.selected_id = list_ids[0]

        hid   = st.session_state.selected_id
        df_wk = backend.get_meter_weekly(df_all, hid)
        frontend.display_meter_annual_signature(df_wk, hid)

    # Moyenne annuelle tous compteurs
    df_yearly = backend.get_yearly_avg(df_all)
    frontend.display_yearly_avg(df_yearly)

    # Heatmap charge
    heat_matrix = backend.get_heatmap_data(df_all)
    frontend.display_load_heatmap(heat_matrix)

    frontend.display_synthesis()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    frontend.display_clustering_header()

    with st.expander("Features utilisees", expanded=False):
        st.markdown("""
| Feature | Description |
|---|---|
| `conso_moy` | Puissance moyenne |
| `conso_std` | Variabilite de la consommation |
| `max_hiver` | Moyenne Nov–Mar (pic chauffage) |
| `max_ete` | Moyenne Jun–Aou |
| `ratio_h_e` | Hiver / ete → signature chauffage electrique |
| `peak_morning` | Conso. 6h–9h |
| `peak_evening` | Conso. 18h–22h |
| `off_peak` | Nuit 0h–5h |
| `ratio_we_wd` | Week-end / semaine → taux d'occupation |
        """)

    with st.spinner("Construction des profils compteurs..."):
        feat = backend.build_features(df_work, work_ids)

    if feat.empty:
        st.error("Pas assez de donnees pour construire les features.")
        st.stop()

    from sklearn.preprocessing import StandardScaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat[backend.FEAT_COLS].values)

    # Courbe du coude
    ks, inertias, sil_scores, best_k = backend.run_elbow(X_scaled)
    k = frontend.display_elbow_chart(ks, inertias, sil_scores, best_k)

    # K-Means final + PCA
    feat_clust, rs_cluster, pca_info, summary = backend.run_clustering(feat, k)

    frontend.display_pca_scatter(feat_clust, pca_info, k)
    frontend.display_clustering_summary(summary)

    # Sauvegarde pour la page Generation
    st.session_state['feat_clustered'] = feat_clust
    st.session_state['cluster_rs']     = rs_cluster

    frontend.display_synthesis()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Classification":
    frontend.display_classification_header()

    test_size, seed = frontend.display_classification_controls()

    with st.spinner("Calcul des features et clustering (k=2)..."):
        X_train, X_test, y_train, y_test, n_min = backend.prepare_classification_data(
            df_work, work_ids, test_size=test_size, seed=seed
        )

    if X_train is None:
        st.warning(
            f"Pas assez de donnees pour entrainer un classifieur "
            f"(minimum 4 exemples par classe, seulement {n_min} disponibles). "
            f"Augmentez N_SAMPLE dans backend.py."
        )
        st.stop()

    with st.spinner("Entrainement des modeles..."):
        results = backend.train_classifiers(X_train, X_test, y_train, y_test)

    frontend.display_classification_results(results)
    frontend.display_synthesis()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    frontend.display_forecasting_header()

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            sel_id  = st.selectbox("Compteur", work_ids)
        with c2:
            horizon = st.slider("Horizon de forecast (jours)", 7, 60, 14)

    with st.spinner("Calcul du forecast..."):
        fc = backend.run_forecasting(df_work, sel_id, horizon=horizon)

    if fc is None:
        st.warning("Pas assez de jours disponibles pour ce compteur (minimum 30 jours).")
        st.stop()

    frontend.display_forecasting_metrics(fc)
    frontend.display_forecast_chart(fc, sel_id)
    frontend.display_residuals_chart(fc)
    frontend.display_future_forecast(fc, horizon, sel_id)

    frontend.display_synthesis()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Generation":
    frontend.display_generation_header()

    # Recuperer le clustering ou le recalculer
    if 'feat_clustered' in st.session_state:
        feat_typed = st.session_state['feat_clustered'].copy()
        rs_cluster = st.session_state['cluster_rs']
        feat_typed['type'] = feat_typed['cluster'].apply(
            lambda c: "RS" if c == rs_cluster else "RP"
        )
    else:
        with st.spinner("Calcul des features et clustering (k=2)..."):
            feat_typed, rs_cluster = backend.get_typed_ids(df_work, work_ids)

    res_type, noise_pct, spike_pct, n_sample = frontend.display_generation_controls(list_ids)

    type_key = "RS" if "RS" in res_type else "RP"
    ids_type = feat_typed[feat_typed['type'] == type_key]['id'].tolist()

    if len(ids_type) < 3:
        st.info("Pas assez de compteurs de ce type. Executez d'abord la page Clustering.")
        st.stop()

    if st.button("Generer un nouveau profil"):
        st.session_state.gen_seed = int(np.random.randint(0, 10_000))

    seed_gen = st.session_state.get('gen_seed', 0)
    df_synth = backend.generate_profile(ids_type, df_work, noise_pct, spike_pct, n_sample, seed=seed_gen)

    df_real_avg, _ = backend.get_real_avg_by_type(df_work, feat_typed, type_key)

    frontend.display_generation_chart(df_synth, df_real_avg, res_type, type_key)

    scores = backend.score_generation(df_synth, df_real_avg)
    frontend.display_generation_scores(scores)

    frontend.display_generation_comparison(df_work, feat_typed)
    frontend.display_synthesis()
