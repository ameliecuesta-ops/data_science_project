"""
frontend.py — Tout le rendu Streamlit : layout, CSS, graphiques.
Style clair (#007BFF, fond blanc) issu du main original.
Graphiques riches (elbow, PCA avec loadings, forecast recursif, generation) issus de Gaetan.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Palette claire (main original) ───────────────────────────────────────────
COLORS = {
    'primary'  : '#007BFF',
    'secondary': '#0056D2',
    'orange'   : '#FF7F0E',
    'green'    : '#28A745',
    'red'      : '#DC3545',
    'purple'   : '#6f42c1',
    'grid'     : '#e9ecef',
    'text'     : '#212529',
    'muted'    : '#6c757d',
    'bg'       : '#ffffff',
    'card_bg'  : '#f0f7ff',
    'card_border': '#cce3fd',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor = 'white',
    plot_bgcolor  = 'white',
    font          = dict(color=COLORS['text'], family='Helvetica Neue, Arial, sans-serif'),
    legend        = dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=COLORS['grid'], borderwidth=1),
    margin        = dict(l=40, r=20, t=50, b=40),
)
_XAXIS = dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], linecolor=COLORS['grid'])
_YAXIS = dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], linecolor=COLORS['grid'])

MOIS_LABELS = ['Nov', 'Dec', 'Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct']
MOIS_TICKS  = [i * (52 / 12) for i in range(12)]

# Palette categorielle claire
CAT_COLORS = ['#007BFF', '#FF7F0E', '#28A745', '#DC3545', '#6f42c1', '#fd7e14']


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG PAGE & LAYOUT GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def set_page_config_and_title():
    st.set_page_config(
        page_title="Dashboard Enedis Pro",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg, #007BFF 0%, #0056D2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 10px 20px rgba(0,123,255,0.2);
}
.main-title h1 {
    margin: 0;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: white !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
}
.main-title p {
    margin: 10px 0 0 0;
    font-size: 1.05rem;
    opacity: 0.9;
    font-weight: 300;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] {
    background-color: #f0f7ff;
    border-radius: 12px;
    border: 1px solid #cce3fd;
    padding: 14px 18px;
    box-shadow: 0 2px 6px rgba(0,123,255,0.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    color: #007BFF !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #6c757d !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.stButton > button {
    background: linear-gradient(135deg, #007BFF, #0056D2);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: opacity .2s, transform .15s;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #6c757d !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #007BFF !important;
    border-bottom-color: #007BFF !important;
}
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>

<div class="main-title">
    <h1>⚡ DASHBOARD ANALYSE ENERGETIQUE ENEDIS</h1>
    <p>Exploration · Clustering K-Means · Classification IA · Forecasting · Generation de profils</p>
</div>
""", unsafe_allow_html=True)


def render_sidebar(n_initial, n_sample):
    with st.sidebar:
        st.markdown("""
        <div style='margin-bottom:20px;padding:16px;background:#f0f7ff;border-radius:12px;border:1px solid #cce3fd'>
          <h2 style='color:#007BFF;margin:0;font-size:1.3rem;font-weight:800'>Energy & IA</h2>
          <p style='color:#6c757d;font-size:0.78rem;margin:4px 0 0'>Enedis Open Data · RES2 · 6-9 kVA</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Exploration", "Clustering", "Classification", "Forecasting", "Generation"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("""
        <div style='font-size:0.8rem;color:#6c757d;line-height:1.7'>
        <b style='color:#212529'>Donnees :</b> Enedis Open Data<br>
        <b style='color:#212529'>Resolution :</b> 30 min → puissance × 0.5<br>
        <b style='color:#212529'>Cible :</b> RS (secondaire) vs RP (principale)
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        load_all = st.session_state.get('load_all', False)
        if not load_all:
            st.caption(f"Affichage des **{n_initial}** premiers compteurs.")
            if st.button("Charger tous les compteurs", use_container_width=True):
                st.session_state['load_all'] = True
                st.rerun()
        else:
            st.caption("Tous les compteurs charges.")
            if st.button("Revenir aux 100 premiers", use_container_width=True):
                st.session_state['load_all'] = False
                st.rerun()

    return page, st.session_state.get('load_all', False)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def display_exploration_header():
    st.markdown("<h2 style='color:#007BFF'>📊 Exploration des donnees</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6c757d;max-width:720px'>Apercu du fichier brut, "
        "statistiques globales et signature annuelle de consommation par compteur.</p>",
        unsafe_allow_html=True,
    )


def display_exploration_kpis(kpis):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compteurs charges",   kpis['n_meters'])
    c2.metric("Lectures totales",    f"{kpis['n_readings']:,}".replace(",", " "))
    c3.metric("Conso. moyenne",      f"{kpis['avg_power']:.0f} W")
    c4.metric("Energie totale",      f"{kpis['total_kwh']:,.0f} kWh")


def display_raw_preview(df_prev):
    with st.container(border=True):
        st.subheader("Apercu du fichier brut")
        st.dataframe(
            df_prev,
            use_container_width=True, hide_index=True,
            column_config={
                "valeur"   : st.column_config.NumberColumn("Puissance (W)", format="%.1f"),
                "id"       : st.column_config.TextColumn("ID Compteur"),
                "horodate" : st.column_config.TextColumn("Horodate"),
            }
        )


def display_power_distribution(df_all):
    with st.container(border=True):
        st.subheader("Distribution globale des puissances")
        fig = px.histogram(
            df_all, x='valeur', nbins=80,
            labels={'valeur': 'Puissance (W)', 'count': 'Nombre'},
            color_discrete_sequence=[COLORS['primary']],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Distribution des lectures instantanees (tous compteurs)",
            xaxis=_XAXIS, yaxis=_YAXIS,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_meter_annual_signature(df_wk, meter_id):
    with st.container(border=True):
        st.subheader(f"Signature annuelle — Compteur {meter_id}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_wk['semaine_id'], y=df_wk['valeur'],
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2.5),
            marker=dict(size=5, color=COLORS['secondary']),
            fill='tozeroy', fillcolor='rgba(0,123,255,0.06)',
            name="Conso. hebdo. moy.",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Consommation hebdomadaire — Compteur {meter_id}",
            xaxis=dict(tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS, **_XAXIS),
            yaxis=dict(range=[0, None], title="Puissance (W)", **_YAXIS),
        )
        st.plotly_chart(fig, use_container_width=True)


def display_yearly_avg(df_yearly):
    with st.container(border=True):
        st.subheader("Moyenne annuelle (tous compteurs)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_yearly['semaine_id'], y=df_yearly['valeur'],
            mode='lines', fill='tozeroy',
            line=dict(color=COLORS['primary'], width=2.5),
            fillcolor='rgba(0,123,255,0.08)',
            name='Puissance moy. (W)',
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Puissance hebdomadaire moyenne — ensemble des compteurs",
            xaxis=dict(tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS, title='Mois', **_XAXIS),
            yaxis=dict(range=[0, None], title='Puissance moy. (W)', **_YAXIS),
        )
        st.plotly_chart(fig, use_container_width=True)


def display_load_heatmap(heat_matrix):
    DAY_LABELS = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    with st.container(border=True):
        st.subheader("Heatmap de charge — heure × jour de semaine")
        st.caption("Puissance moyenne (W) par creneau horaire, tous compteurs confondus.")
        fig = go.Figure(go.Heatmap(
            z=heat_matrix.values,
            x=DAY_LABELS,
            y=[f"{h:02d}:00" for h in range(24)],
            colorscale=[
                [0.0,  '#f0f7ff'],
                [0.4,  '#007BFF'],
                [0.75, '#0056D2'],
                [1.0,  '#003494'],
            ],
            colorbar=dict(title='W'),
            hovertemplate='%{x} %{y}<br>%{z:.0f} W<extra></extra>',
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title='Puissance moyenne par heure et jour de semaine',
            xaxis=dict(side='top', gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def display_clustering_header():
    st.markdown("<h2 style='color:#007BFF'>🔵 Clustering · K-Means</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6c757d;max-width:720px'>"
        "Detection automatique des <b>Residences Principales (RP)</b> et "
        "<b>Residences Secondaires (RS)</b> par regroupement de compteurs "
        "aux comportements similaires.</p>",
        unsafe_allow_html=True,
    )


def display_elbow_chart(ks, inertias, sil_scores, best_k):
    with st.container(border=True):
        st.subheader("Combien de groupes ? — Methode du coude")
        st.markdown(
            "<p style='color:#6c757d;font-size:0.88rem;max-width:760px;margin-bottom:12px'>"
            "L'<b style='color:#007BFF'>inertie</b> (axe gauche) mesure la compacite des clusters. "
            "Le <b>coude</b> indique ou l'ajout de groupes apporte peu. "
            "Le <b style='color:#FF7F0E'>score silhouette</b> (axe droit, 0→1) confirme "
            "le k optimal : plus il est eleve, mieux les clusters sont separes."
            "</p>", unsafe_allow_html=True,
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=ks, y=inertias, name="Inertie",
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2.5),
            marker=dict(size=8),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=ks, y=sil_scores, name="Score Silhouette",
            mode='lines+markers',
            line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
            marker=dict(size=8),
        ), secondary_y=True)
        fig.add_vline(x=best_k, line_dash='dot', line_color=COLORS['green'], line_width=2)
        fig.add_annotation(
            x=best_k, y=inertias[ks.index(best_k)], xref='x', yref='y',
            text=f"  k optimal = {best_k}", showarrow=False,
            font=dict(color=COLORS['green'], size=13), xanchor='left',
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Courbe du coude & Score Silhouette")
        fig.update_yaxes(title_text="Inertie ↓ (compacite)", secondary_y=False,  gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        fig.update_yaxes(title_text="Silhouette ↑ (separation)", secondary_y=True, gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        fig.update_xaxes(title_text="Nombre de groupes (k)", gridcolor=COLORS['grid'], tickmode='linear', dtick=1)
        st.plotly_chart(fig, use_container_width=True)

        col_k, col_hint, _ = st.columns([1, 2, 1])
        with col_k:
            k = st.slider("Nombre de groupes (k)", 2, 6, min(best_k, 6))
        with col_hint:
            st.markdown(
                f"<p style='color:#6c757d;font-size:0.82rem;margin-top:28px'>"
                f"Suggere par silhouette : <b style='color:#28A745'>k = {best_k}</b>. "
                f"Vous pouvez ajuster avec le slider.</p>",
                unsafe_allow_html=True,
            )
        return k


def display_pca_scatter(feat, pca_info, k):
    palette = {f"Groupe {i+1}": CAT_COLORS[i] for i in range(k)}
    var1, var2 = pca_info['var1'], pca_info['var2']
    top_pc1, top_pc2 = pca_info['top_pc1'], pca_info['top_pc2']

    with st.container(border=True):
        st.subheader("Carte des compteurs — qui ressemble a qui ?")
        st.caption(
            "Chaque point est un compteur. Les points proches ont des profils similaires. "
            "Les couleurs indiquent le groupe detecte automatiquement."
        )
        fig = px.scatter(
            feat, x='PC1', y='PC2',
            color='cluster_label', symbol='type',
            hover_data=['id', 'conso_moy', 'ratio_h_e', 'ratio_we_wd'],
            color_discrete_map=palette,
            labels={
                'PC1': f"PC1 ({var1:.1f}% variance) — principalement : {', '.join(top_pc1)}",
                'PC2': f"PC2 ({var2:.1f}% variance) — principalement : {', '.join(top_pc2)}",
                'cluster_label': 'Groupe', 'type': 'Type detecte',
            },
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Compteurs groupes par profil de consommation")
        fig.update_traces(marker=dict(size=10, opacity=0.80))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<p style='color:#6c757d;font-size:0.82rem;margin-top:-8px'>"
            f"<b style='color:#007BFF'>PC1</b> ({var1:.1f}%) : principalement <code>{', '.join(top_pc1)}</code>. "
            f"<b style='color:#007BFF'>PC2</b> ({var2:.1f}%) : principalement <code>{', '.join(top_pc2)}</code>. "
            f"Ensemble : <b>{var1+var2:.1f}%</b> de la variance totale.</p>",
            unsafe_allow_html=True,
        )


def display_clustering_summary(summary):
    with st.container(border=True):
        st.subheader("Tableau recapitulatif des groupes")
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def display_classification_header():
    st.markdown("<h2 style='color:#007BFF'>🤖 Classification · RS vs RP</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6c757d;max-width:720px'>"
        "Entrainement et comparaison de deux modeles supervises (Regression Logistique et MLP) "
        "pour predire RP vs RS a partir des features de consommation. "
        "Les labels proviennent du clustering K-Means.</p>",
        unsafe_allow_html=True,
    )


def display_classification_controls():
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Taille du jeu de test (%)", 20, 40, 30) / 100
        with c2:
            seed = st.slider("Graine aleatoire", 0, 99, 42)
    return test_size, seed


def display_classification_results(results):
    clr_list = [COLORS['primary'], COLORS['orange']]
    tabs     = st.tabs(list(results.keys()))

    for i, (name, res) in enumerate(results.items()):
        with tabs[i]:
            rpt = res['report']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",       f"{rpt['accuracy']*100:.1f}%")
            c2.metric("Precision RS",   f"{rpt['RS']['precision']*100:.1f}%")
            c3.metric("Recall RS",      f"{rpt['RS']['recall']*100:.1f}%")
            c4.metric("F1-score RS",    f"{rpt['RS']['f1-score']*100:.1f}%")

            col_a, col_b = st.columns(2)

            with col_a:
                cm   = res['cm']
                labs = ['RP', 'RS']
                fig_cm = px.imshow(
                    cm, text_auto=True, x=labs, y=labs,
                    color_continuous_scale=[[0, '#f0f7ff'], [1, clr_list[i]]],
                    labels={'x': 'Predit', 'y': 'Reel'},
                )
                fig_cm.update_layout(
                    **PLOTLY_LAYOUT, title="Matrice de confusion",
                    xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_b:
                df_imp = res['df_imp']
                fig_imp = px.bar(
                    df_imp, x='importance', y='feature', orientation='h',
                    color='importance',
                    color_continuous_scale=['#f0f7ff', clr_list[i]],
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                )
                fig_imp.update_layout(
                    **PLOTLY_LAYOUT, title="Importance des features",
                    coloraxis_showscale=False, xaxis=_XAXIS,
                    yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("Rapport de classification complet"):
                st.code(res['report_txt'])

            with st.expander("ℹ Methodologie & limites"):
                st.markdown("""
**Ce que fait ce classifieur**

Le modele apprend a predire si un compteur est une *Residence Principale (RP)* ou
*Secondaire (RS)* a partir de 9 features (moyennes, pics, ratios).

**Comment les labels sont generes**

Les labels viennent du clustering K-Means (k=2) : le cluster avec le ratio
week-end/semaine le plus eleve est appele RS. Le classifieur re-apprend donc
la frontiere du clustering de facon supervisee.

**Pourquoi c'est quand meme utile**
- Revele quelles features comptent le plus pour la distinction RS/RP.
- Un classifieur entraine peut labelliser de nouveaux compteurs en O(1) sans relancer K-Means.
- Comparer LogReg vs MLP montre si la frontiere est lineaire ou complexe.

**Limites**
- Haute accuracy ≠ clustering correct : le modele reproduit les decisions du clustering.
- Avec ~50 compteurs, le jeu de test est tres petit (~15 compteurs). Testez differentes graines.
- Une evaluation rigoureuse necessite des labels terrain (contrats Enedis RP/RS).
                """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def display_forecasting_header():
    st.markdown("<h2 style='color:#007BFF'>📈 Prevision de consommation</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6c757d;max-width:720px'>"
        "Prediction de la consommation journaliere future par regression lineaire avec features "
        "de lags et de saisonnalite, plus une baseline moyenne mobile de reference.</p>",
        unsafe_allow_html=True,
    )


def display_forecasting_metrics(fc):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (modele)",    f"{fc['mae_lr']:.1f} W")
    c2.metric("RMSE (modele)",   f"{fc['rmse_lr']:.1f} W")
    c3.metric("R² (modele)",     f"{fc['r2_lr']:.3f}")
    c4.metric("MAE (baseline)", f"{fc['mae_base']:.1f} W",
              delta=f"{fc['mae_base']-fc['mae_lr']:+.1f} W vs modele",
              delta_color="inverse")


def display_forecast_chart(fc, meter_id):
    with st.container(border=True):
        df_day     = fc['df_day']
        test_dates = fc['test_dates']
        y_te       = fc['y_te']
        y_pred     = fc['y_pred']
        y_base     = fc['y_base']
        rmse_lr    = fc['rmse_lr']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_day['date'].astype(str).tolist(), y=df_day['valeur'],
            mode='lines', name='Historique',
            line=dict(color=COLORS['grid'], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_te,
            mode='lines', name='Reel (test)',
            line=dict(color=COLORS['primary'], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred,
            mode='lines', name='Prediction modele',
            line=dict(color=COLORS['orange'], width=2, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_base,
            mode='lines', name='Baseline (moy. mobile)',
            line=dict(color=COLORS['purple'], width=1.5, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=test_dates + test_dates[::-1],
            y=list(y_pred + rmse_lr) + list((y_pred - rmse_lr)[::-1]),
            fill='toself', fillcolor='rgba(255,127,14,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='Intervalle ±RMSE',
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Prevision journaliere — Compteur {meter_id}",
            xaxis=dict(**_XAXIS, title="Date"),
            yaxis=dict(**_YAXIS, title="Puissance moy. journaliere (W)"),
        )
        st.plotly_chart(fig, use_container_width=True)


def display_residuals_chart(fc):
    with st.container(border=True):
        residus = fc['y_te'] - fc['y_pred']
        fig = px.histogram(
            residus, nbins=30,
            color_discrete_sequence=[COLORS['green']],
            labels={'value': 'Erreur (W)', 'count': 'Nombre'},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Erreurs de prediction — distribution des residus",
            xaxis=_XAXIS, yaxis=_YAXIS,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_future_forecast(fc, horizon, meter_id):
    with st.container(border=True):
        st.subheader(f"Forecast futur — {horizon} prochains jours")
        st.caption("Chaque etape future est predite a partir des precedentes (forecast recursif).")

        ctx_dates    = fc['ctx_dates']
        ctx_vals     = fc['ctx_vals']
        future_dates = fc['future_dates']
        future_preds = fc['future_preds']
        rmse_lr      = fc['rmse_lr']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ctx_dates, y=ctx_vals,
            mode='lines', name='Historique recent',
            line=dict(color=COLORS['primary'], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[ctx_dates[-1]] + future_dates,
            y=[ctx_vals[-1]]  + future_preds,
            mode='lines+markers', name=f'Forecast ({horizon} jours)',
            line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=[v + rmse_lr for v in future_preds] + [max(v - rmse_lr, 0) for v in future_preds[::-1]],
            fill='toself', fillcolor='rgba(255,127,14,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='Bande ±RMSE',
        ))
        vline_x = pd.to_datetime(ctx_dates[-1]).timestamp() * 1000
        fig.add_vline(
            x=vline_x, line_dash='dot', line_color=COLORS['green'],
            annotation_text="Aujourd'hui", annotation_position='top right',
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Forecast de consommation future — Compteur {meter_id}",
            xaxis=dict(**_XAXIS, title="Date"),
            yaxis=dict(**_YAXIS, title="Puissance moy. journaliere (W)", rangemode='tozero'),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def display_generation_header():
    st.markdown("<h2 style='color:#007BFF'>🔮 Generation de profils synthetiques (RP/RS)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6c757d;max-width:720px'>"
        "Generation d'un profil annuel realiste conditionne au type de residence (RS ou RP). "
        "Du bruit aleatoire et des pics occasionnels imitent la variabilite reelle. "
        "Le profil synthetique est compare aux donnees reelles pour verifier la coherence.</p>",
        unsafe_allow_html=True,
    )


def display_generation_controls(list_ids):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            res_type  = st.selectbox("Type de residence", ["RP (principale)", "RS (secondaire)"])
        with c2:
            noise_pct = st.slider("Bruit aleatoire (%)", 5, 40, 20)
        with c3:
            spike_pct = st.slider("Frequence des pics (%)", 0, 20, 7)
        with c4:
            n_sample  = st.slider("Compteurs de reference", 5, min(20, len(list_ids)), 10)
    return res_type, noise_pct, spike_pct, n_sample


def display_generation_chart(df_synth, df_real_avg, res_type, type_key):
    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_real_avg['semaine_id'], y=df_real_avg['valeur'],
            mode='lines', name=f'Reel moyen ({type_key})',
            line=dict(color=COLORS['primary'], width=2, dash='dot'),
        ))
        fig.add_trace(go.Scatter(
            x=df_synth['semaine_id'], y=df_synth['synth'],
            mode='lines+markers', name=f'Profil synthetique ({type_key})',
            line=dict(color=COLORS['orange'], width=2.5),
            marker=dict(size=5),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Synthetique vs reel — {res_type}",
            xaxis=dict(tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS, title='Mois', **_XAXIS),
            yaxis=dict(range=[0, None], title='Puissance (W)', **_YAXIS),
        )
        st.plotly_chart(fig, use_container_width=True)


def display_generation_scores(scores):
    if scores:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE vs reel",       f"{scores['mae']:.1f} W")
        c2.metric("RMSE vs reel",      f"{scores['rmse']:.1f} W")
        c3.metric("Similarite R²",     f"{scores['r2']:.3f}")


def display_generation_comparison(df_work, feat_typed):
    with st.container(border=True):
        st.subheader("RP vs RS — profils reels moyens cote a cote")

        rp_ids = feat_typed[feat_typed['type'] == 'RP']['id'].tolist()
        rs_ids = feat_typed[feat_typed['type'] == 'RS']['id'].tolist()

        rp_avg = df_work[df_work['id'].isin(rp_ids)].groupby('semaine_id')['valeur'].mean().reset_index()
        rs_avg = df_work[df_work['id'].isin(rs_ids)].groupby('semaine_id')['valeur'].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rp_avg['semaine_id'], y=rp_avg['valeur'],
            mode='lines', name='Moy. RP',
            line=dict(color=COLORS['primary'], width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=rs_avg['semaine_id'], y=rs_avg['valeur'],
            mode='lines', name='Moy. RS',
            line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Profils hebdomadaires moyens — RP vs RS",
            xaxis=dict(tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS, **_XAXIS),
            yaxis=dict(range=[0, None], title='Puissance (W)', **_YAXIS),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Insight cle** : une RS presente generalement une conso. plus elevee le week-end "
        "et pendant les vacances, tandis qu'une RP a une utilisation quotidienne stable "
        "avec un pic hivernal marque (chauffage electrique RES2)."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHESE
# ══════════════════════════════════════════════════════════════════════════════

def display_synthesis():
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Synthese Methodologique")
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Preparation des donnees**")
                st.caption(
                    "• Format long → features par compteur\n"
                    "• Conversion puissance × 0.5 → energie\n"
                    "• Echantillon reproductible (50 compteurs ML)"
                )
            with c2:
                st.markdown("**Modeles**")
                st.caption(
                    "• K-Means (labellisation RS/RP)\n"
                    "• Comparaison LogReg vs MLP\n"
                    "• Regression lineaire avec lags (forecast)"
                )
            with c3:
                st.markdown("**Metriques**")
                st.caption(
                    "• Accuracy, Precision, Recall, F1 (classification)\n"
                    "• MAE, RMSE, R² (regression/forecast)\n"
                    "• Silhouette score (clustering)"
                )
