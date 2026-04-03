import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def set_page_config_and_title():
    st.set_page_config(page_title="Dashboard Enedis Pro", layout="wide")
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
        .mon-bloc-bleu {
            background-color: #f0f7ff !important; 
            border-radius: 12px !important;
            border: 1px solid #cce3fd !important; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03) !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
        }
        div[data-testid="stMetric"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        </style>
        
        <div class="main-title">
            <h1>DASHBOARD ANALYSE ÉNERGÉTIQUE ENEDIS</h1>
            <p>Analyse de consommation, Clustering K-Means, Prédictions IA et Simulation Panneaux Solaires</p>
        </div>
        """, unsafe_allow_html=True)

def display_kpi_section(val_max, val_moyenne, nb_clients):
    str_max = f"{val_max:,.0f}".replace(",", " ") + " kWh"
    str_moy = f"{val_moyenne:,.0f}".replace(",", " ") + " kWh"
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

def display_classification_tab(acc, cm):
    st.metric("Précision (Accuracy)", f"{acc*100:.1f}%")
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prédit", y="Réel"),
                       x=['Type A', 'Type B'], y=['Type A', 'Type B'],
                       color_continuous_scale='Blues', height=350)
    fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_cm, width='stretch')

def display_regression_tab(mae_rf, x_range, y_range_preds, df_sample_graph):
    st.metric("Erreur Moyenne (MAE)", f"{mae_rf:.1f} kWh")
    fig_reg = px.scatter(
        df_sample_graph, x='conso_pointe_soir', y='max', color='Profil',
        trendline="ols", title="Prédiction du Pic via Random Forest (Non-linéaire)",
        color_discrete_sequence=['#007BFF', '#FF7F0E'], height=350
    )
    fig_reg.add_trace(go.Scatter(
        x=x_range.flatten(), y=y_range_preds, mode='lines', 
        name='Modèle RF', line=dict(color='black', width=2)
    ))
    fig_reg.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_reg, width='stretch')

def display_client_curve(client_data, conso_cols, selected_id):
    fig = px.line(
        x=conso_cols, y=client_data.values, height=500,
        title=f"Consommation du client {selected_id}",
        labels={'x': 'Date / Heure', 'y': 'Énergie (kWh)'}
    )
    fig.update_traces(line=dict(width=1)) 
    fig.update_xaxes(
        rangeslider_visible=True, rangeslider_thickness=0.06,
        rangeselector=dict(
            y=0.98, x=0.01, yanchor="top", xanchor="left", bgcolor="rgba(230,230,230,0.8)",
            buttons=list([
                dict(count=7, label="1 Sem", step="day", stepmode="backward"),
                dict(count=1, label="1 Mois", step="month", stepmode="backward"),
                dict(step="all", label="Tout")
            ])
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

def display_client_profile_info(type_client, explication, couleur_bord, conso_nuit, conso_journee, conso_soiree):
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 6px solid {couleur_bord}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h3 style="margin: 0 0 5px 0; color: {couleur_bord}; font-size: 1.4rem;">{type_client}</h3>
        <p style="margin: 0; font-size: 0.95rem; color: #333;">{explication}</p>
        <hr style="margin: 10px 0; opacity: 0.1;">
        <p style="margin: 0; font-size: 0.85rem; color: #666;">
            <b>Analyse des habitudes :</b> Moyenne Nuit = {conso_nuit:.1f} kW | Moyenne Journée = {conso_journee:.1f} kW | Moyenne Soirée = {conso_soiree:.1f} kW
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_detailed_analysis(selected_id, client_data, df, conso_cols):
    st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)
    st.markdown(f"**Comparaison de la distribution (Moyenne, Médiane, Min/Max) : Global vs Client {selected_id}**")
    
    global_values = df[conso_cols].values.flatten()
    if len(global_values) > 100000:
        np.random.seed(42)
        global_values = np.random.choice(global_values, 100000, replace=False)
        
    fig_dist = ff.create_distplot([global_values], ['Distribution Globale (Tous les clients)'], show_hist=True, show_rug=False, colors=['#1f77b4'])
    conso_moyenne_ce_client = client_data.mean()
    
    fig_dist.add_vline(x=conso_moyenne_ce_client, line_width=3, line_dash="dash", line_color="#FF4B4B")
    fig_dist.add_annotation(
        x=conso_moyenne_ce_client, y=0.9, yref="paper",
        text=f"Moyenne Client {selected_id} : {conso_moyenne_ce_client:.0f} kWh",
        showarrow=True, arrowhead=1, arrowcolor="#FF4B4B", arrowsize=1.5,
        ax=80, ay=-30, bgcolor="#FF4B4B", font=dict(color="white", size=12)
    )
    fig_dist.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_title="Consommation (kWh)", yaxis_title="Densité de clients")
    st.plotly_chart(fig_dist, use_container_width=True)

def display_solar_simulation(df_simul, client_data):
    fig_simul = px.line(
        df_simul, x='Date', y=['Sans Solaire (Avant)', 'Avec Solaire (Après)'], 
        color_discrete_sequence=['#1f77b4', '#ff7f0e'], height=350,
        labels={'value': 'Énergie (kWh)', 'Date': 'Date'}
    )
    fig_simul.update_traces(line=dict(width=2), selector=dict(name='Sans Solaire (Avant)'))
    fig_simul.update_traces(line=dict(width=3), selector=dict(name='Avec Solaire (Après)'))
    
    date_debut = df_simul['Date'].iloc[0]
    date_fin = pd.to_datetime(date_debut) + pd.Timedelta(days=7)

    fig_simul.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, client_data.values.max() * 1.1]) 
    )
    
    fig_simul.update_xaxes(
        rangeslider_visible=True, rangeslider_thickness=0.08, 
        range=[date_debut, date_fin.strftime('%Y-%m-%d %H:%M:%S')],
        rangeselector=dict(
            y=1.1, x=0.01, yanchor="top", xanchor="left", bgcolor="rgba(230,230,230,0.8)",
            buttons=list([
                dict(count=7, label="1 Sem", step="day", stepmode="backward"),
                dict(count=1, label="1 Mois", step="month", stepmode="backward"),
                dict(step="all", label="Tout")
            ])
        )
    )
    st.plotly_chart(fig_simul, use_container_width=True)

def display_synthesis():
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

def display_pca_chart(df_pca, labels_sample):
    df_pca['Profil'] = labels_sample
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Profil', title="PCA : Séparation des profils")
    st.plotly_chart(fig, use_container_width=True)

def display_forecasting(y_reel, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_reel[-48:], name="Réel"))
    fig.add_trace(go.Scatter(y=y_pred[-48:], name="Prédiction", line=dict(dash='dash', color='red')))
    st.plotly_chart(fig, use_container_width=True)

def display_generator_section(t, values, type_nom):
    fig = px.area(x=t, y=values, title=f"Simulation : {type_nom}")
    st.plotly_chart(fig, use_container_width=True)
