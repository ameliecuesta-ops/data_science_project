"""
Residential Energy Data Analysis Dashboard
===========================================
Project: clustering, classification, forecasting and generation
of consumption profiles using Enedis open data (RES2 6-9kVA).

Modules:
  1. Exploration    – raw data overview and annual signature
  2. Clustering     – K-Means to detect Secondary vs Primary residences
  3. Classification – Logistic Regression / Neural Network
  4. Forecasting    – Linear Regression / moving average baseline
  5. Generation     – synthetic profile generator (SR / PR)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    silhouette_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Energy Data Analysis · Enedis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Thème CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}
code, .stCode { font-family: 'DM Mono', monospace !important; }

/* ── Background ── */
.stApp { background: #0b0e17; color: #e8eaf2; }
section[data-testid="stSidebar"] { background: #10141f !important; border-right: 1px solid #1e2535; }

/* ── Cards ── */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #13182a;
    border: 1px solid #1e2840;
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1a2035;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #242d45;
}
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 2rem !important; color: #7ee8fa !important; }
[data-testid="stMetricLabel"] { color: #8892aa !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #382ecc, #5b4aff);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: .04em;
    padding: 10px 24px;
    transition: opacity .2s, transform .15s;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }

/* ── Selectbox / Slider ── */
[data-testid="stSelectbox"] > div, [data-testid="stSlider"] { color: #e8eaf2; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: .06em;
    color: #8892aa !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #7ee8fa !important; border-bottom-color: #7ee8fa !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Info / Warning ── */
.stAlert { border-radius: 10px; }

/* ── Headings accent ── */
.accent { color: #7ee8fa; }
.orange { color: #f4a261; }
.green  { color: #52d9a0; }
.purple { color: #b07ef4; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS / LOADERS
# ══════════════════════════════════════════════════════════════════════════════
FILE_PATH  = "export.csv"
N_INITIAL = 100    # meters loaded on first start
N_SAMPLE  = 50     # meters used for heavy ML tasks


@st.cache_data(show_spinner=False)
def get_preview(path, skip=0):
    try:
        if skip == 0:
            return pd.read_csv(path, nrows=5)
        return pd.read_csv(path, skiprows=range(1, skip + 1), nrows=5)
    except Exception:
        return pd.read_csv(path, nrows=5)


@st.cache_resource(show_spinner=False)
def load_houses(path, n=None):
    """Load the first n meters (or all meters if n is None)."""
    reader    = pd.read_csv(path, chunksize=100_000)
    data_list = []
    found_ids = []          # list to preserve insertion order

    for chunk in reader:
        for idx in chunk['id'].unique():
            if idx not in found_ids:
                found_ids.append(idx)
        # If we hit the limit, stop after this chunk
        if n is not None and len(found_ids) >= n:
            data_list.append(chunk[chunk['id'].isin(found_ids[:n])])
            break
        data_list.append(chunk)

    df = pd.concat(data_list, ignore_index=True)
    if n is not None:
        found_ids = found_ids[:n]
        df = df[df['id'].isin(found_ids)]

    # ── Timestamp parsing ─────────────────────────────────────────────────────
    df['horodate'] = pd.to_datetime(df['horodate'], utc=True, errors='coerce')
    df = df.dropna(subset=['horodate'])
    df['mois']      = df['horodate'].dt.month
    df['jour']      = df['horodate'].dt.day
    df['heure']     = df['horodate'].dt.hour
    df['dow']       = df['horodate'].dt.dayofweek   # 0=Mon … 6=Sun
    # Fiscal year starting in November: Nov=0, Dec=1, ..., Oct=11
    df['mois_fiscal'] = (df['mois'] - 11) % 12
    df['semaine_id']  = (df['mois_fiscal'] * 4 + (df['jour'] - 1) // 7).clip(0, 51)

    return df, found_ids


def build_features(df_all, ids):
    """
    Build the per-meter feature matrix for clustering / classification.

    Features per household:
      - conso_moy     : average power draw (W)
      - conso_std     : standard deviation
      - max_hiver     : average Nov–Mar (winter)
      - max_ete       : average Jun–Aug (summer)
      - ratio_h_e     : winter / summer → electric heating signature
      - peak_morning  : consumption 6am–9am
      - peak_evening  : consumption 6pm–10pm
      - off_peak      : consumption midnight–5am
      - ratio_we_wd   : weekend vs weekday
    """
    rows = []
    for hid in ids:
        d = df_all[df_all['id'] == hid]
        if len(d) < 50:
            continue

        conso_moy   = d['valeur'].mean()
        conso_std   = d['valeur'].std()
        hiver       = d[d['mois'].isin([11, 12, 1, 2, 3])]['valeur'].mean()
        ete         = d[d['mois'].isin([6, 7, 8])]['valeur'].mean()
        ratio_h_e   = hiver / ete if ete > 0 else np.nan

        peak_morn   = d[d['heure'].between(6, 9)]['valeur'].mean()
        peak_even   = d[d['heure'].between(18, 22)]['valeur'].mean()
        off_peak    = d[d['heure'].between(0, 5)]['valeur'].mean()

        dow         = d['horodate'].dt.dayofweek
        we          = d[dow >= 5]['valeur'].mean()
        wd          = d[dow < 5]['valeur'].mean()
        ratio_we_wd = we / wd if wd > 0 else np.nan

        rows.append({
            'id'           : hid,
            'conso_moy'    : conso_moy,
            'conso_std'    : conso_std,
            'max_hiver'    : hiver,
            'max_ete'      : ete,
            'ratio_h_e'    : ratio_h_e,
            'peak_morning' : peak_morn,
            'peak_evening' : peak_even,
            'off_peak'     : off_peak,
            'ratio_we_wd'  : ratio_we_wd,
        })

    feat = pd.DataFrame(rows).dropna()
    return feat


# ── Plotly colors ─────────────────────────────────────────────────────────────
COLORS = {
    'primary'  : '#7ee8fa',
    'secondary': '#5b4aff',
    'orange'   : '#f4a261',
    'green'    : '#52d9a0',
    'purple'   : '#b07ef4',
    'bg'       : '#13182a',
    'grid'     : '#1e2840',
    'text'     : '#e8eaf2',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor = COLORS['bg'],
    plot_bgcolor  = COLORS['bg'],
    font          = dict(color=COLORS['text'], family='DM Sans'),
    legend        = dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['grid']),
    margin        = dict(l=40, r=20, t=50, b=40),
)

# Default axes — reuse individually
_XAXIS = dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
_YAXIS = dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])

MOIS_LABELS  = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
MOIS_TICKS   = [i * (52 / 12) for i in range(12)]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:24px'>
      <h2 style='color:#7ee8fa;margin:0;font-family:Syne,sans-serif;font-size:1.4rem'>Energy & AI</h2>
      <p style='color:#8892aa;font-size:0.78rem;margin:4px 0 0'>Enedis RES2 · 6-9 kVA</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Exploration", "Clustering", "Classification", "Forecasting", "Generation"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style='color:#8892aa;font-size:0.74rem;line-height:1.6'>
    <b style='color:#e8eaf2'>Data :</b> Enedis Open Data<br>
    <b style='color:#e8eaf2'>Resolution :</b> 30 min → energy × 0.5<br>
    <b style='color:#e8eaf2'>Target :</b> SR (secondary) vs PR (primary)
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    _load_all = st.session_state.get('load_all', False)
    if not _load_all:
        st.caption(f"Currently showing first **{N_INITIAL}** meters.")
        if st.button("Load all meters", use_container_width=True):
            st.session_state['load_all'] = True
            st.rerun()
    else:
        st.caption("All meters loaded.")
        if st.button("Reload first 100 only", use_container_width=True):
            st.session_state['load_all'] = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
data_ok = os.path.exists(FILE_PATH)

@st.cache_resource(show_spinner=False)
def get_work_sample(path, n, sample_n, seed=42):
    """Cache the work sample slice so it is never recomputed on reruns."""
    df, ids = load_houses(path, n=n)
    rng      = np.random.default_rng(seed)
    w_ids    = list(rng.choice(ids, size=min(sample_n, len(ids)), replace=False))
    df_w     = df[df['id'].isin(w_ids)].copy()
    return df, ids, df_w, w_ids

if data_ok:
    _load_all = st.session_state.get('load_all', False)
    _n        = None if _load_all else N_INITIAL
    # Returns instantly from cache on all reruns — no spinner flash.
    df_all, list_ids, df_work, work_ids = get_work_sample(FILE_PATH, _n, N_SAMPLE)
else:
    st.error(f"⚠️ File `{FILE_PATH}` not found. Place it at the project root.", icon="🚫")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Exploration":
    st.markdown("<h1>Data Exploration</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892aa;max-width:720px'>Overview of the raw file, "
        "general statistics and annual consumption signature per meter.</p>",
        unsafe_allow_html=True
    )

    # ── KPIs — full loaded dataset ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Meters loaded", len(list_ids))
    with col2:
        st.metric("Total readings", f"{len(df_all):,}".replace(",", " "))
    with col3:
        st.metric("Avg. consumption", f"{df_all['valeur'].mean():.0f} W")
    with col4:
        total_kwh = df_all['valeur'].sum() * 0.5 / 1000
        st.metric("Total energy", f"{total_kwh:,.0f} kWh")

    _load_all = st.session_state.get('load_all', False)
    if not _load_all:
        st.caption(
            f"Showing first **{len(list_ids)} meters**. "
            f"Use **Load all meters** in the sidebar to include the full dataset. "
            f"ML pages (Clustering, Classification, Forecasting, Generation) use a "
            f"reproducible sample of **{len(work_ids)} meters**."
        )
    elif len(list_ids) > N_SAMPLE:
        st.caption(
            f"Stats reflect all **{len(list_ids)} meters** loaded. "
            f"ML pages use a reproducible sample of **{len(work_ids)} meters**."
        )

    st.divider()

    # ── Aperçu CSV ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Raw file sample")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("5 random rows"):
                st.session_state.skip_rows = int(np.random.randint(1, 50_000))
        skip = st.session_state.get('skip_rows', 0)
        df_prev = get_preview(FILE_PATH, skip=skip)
        st.dataframe(
            df_prev,
            use_container_width=True, hide_index=True,
            column_config={
                "valeur"   : st.column_config.NumberColumn("Power (W)", format="%.1f"),
                "id"       : st.column_config.TextColumn("Meter ID"),
                "horodate" : st.column_config.TextColumn("Timestamp"),
            }
        )

    # ── Distribution globale ───────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Overall power distribution")
        fig_hist = px.histogram(
            df_all, x='valeur', nbins=80,
            labels={'valeur': 'Power (W)', 'count': 'Count'},
            color_discrete_sequence=[COLORS['primary']],
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, title="Distribution of instantaneous power readings (all meters)",
                               xaxis=_XAXIS, yaxis=_YAXIS)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Signature annuelle ────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Annual consumption signature of a meter")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("Another meter"):
                st.session_state.selected_id = np.random.choice(list_ids)
        if 'selected_id' not in st.session_state:
            st.session_state.selected_id = list_ids[0]

        hid    = st.session_state.selected_id
        df_h   = df_all[df_all['id'] == hid].copy()
        df_wk  = df_h.groupby('semaine_id')['valeur'].mean().reset_index()

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=df_wk['semaine_id'], y=df_wk['valeur'],
            mode='lines+markers',
            line=dict(color=COLORS['secondary'], width=2.5),
            marker=dict(size=5, color=COLORS['primary']),
            name="Weekly avg. consumption"
        ))
        fig_h.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Weekly consumption — Meter {hid}",
            xaxis=dict(
                tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS,
                gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']
            ),
            yaxis=dict(range=[0, None], title="Power (W)", gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Yearly average across all meters ─────────────────────────────────────
    with st.container(border=True):
        st.subheader("Yearly average consumption (all meters)")
        df_yearly = (
            df_all.groupby('semaine_id')['valeur']
            .mean().reset_index()
            .sort_values('semaine_id')
        )
        fig_yr = go.Figure()
        fig_yr.add_trace(go.Scatter(
            x=df_yearly['semaine_id'], y=df_yearly['valeur'],
            mode='lines', fill='tozeroy',
            line=dict(color=COLORS['primary'], width=2),
            fillcolor='rgba(126,232,250,0.10)',
            name='Avg. power (W)',
        ))
        fig_yr.update_layout(
            **PLOTLY_LAYOUT,
            title="Weekly average power — all meters combined",
            xaxis=dict(
                tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS,
                title='Month', gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']
            ),
            yaxis=dict(range=[0, None], title='Avg. power (W)',
                       gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    # ── Weekly load heatmap: hour × day-of-week ───────────────────────────────
    with st.container(border=True):
        st.subheader("Load heatmap — hour of day × day of week")
        st.caption("Average power (W) for each hour slot across all meters and all weeks.")

        DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df_heat = (
            df_all.groupby(['dow', 'heure'])['valeur']
            .mean().reset_index()
        )
        # Pivot to matrix: rows = hours (0-23), cols = days (0-6)
        heat_matrix = (
            df_heat.pivot(index='heure', columns='dow', values='valeur')
            .reindex(index=range(24), columns=range(7))
        )

        fig_heat = go.Figure(go.Heatmap(
            z=heat_matrix.values,
            x=DAY_LABELS,
            y=[f"{h:02d}:00" for h in range(24)],
            colorscale=[
                [0.0,  '#ffffff'],
                [0.35, COLORS['primary']],
                [0.7,  COLORS['secondary']],
                [1.0,  '#0b0e17'],
            ],
            colorbar=dict(title='W', tickfont=dict(color=COLORS['text'])),
            hovertemplate='%{x} %{y}<br>%{z:.0f} W<extra></extra>',
        ))
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            title='Average power by hour and day of week',
            xaxis=dict(side='top', gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(autorange='reversed', gridcolor='rgba(0,0,0,0)'),
            height=520,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.markdown("<h1>Clustering · K-Means</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892aa;max-width:720px'>"
        "Automatic detection of <b>Primary Residences (PR)</b> and "
        "<b>Secondary Residences (SR)</b> by grouping meters with similar "
        "consumption behaviour.</p>",
        unsafe_allow_html=True
    )

    with st.expander("Features used", expanded=False):
        st.markdown("""
        | Feature | Description |
        |---|---|
        | `avg_consumption` | Average power draw |
        | `variability` | How much consumption fluctuates |
        | `winter_avg` | Average Nov–Mar (heating peak) |
        | `summer_avg` | Average Jun–Aug |
        | `winter_summer_ratio` | Winter vs summer → electric heating signature |
        | `morning_peak` | Consumption 6am–9am |
        | `evening_peak` | Consumption 6pm–10pm |
        | `night_offpeak` | Night 12am–5am |
        | `weekend_weekday_ratio` | Weekend vs weekday → occupancy pattern |
        """)

    with st.spinner("Building meter profiles…"):
        feat = build_features(df_work, work_ids)

    if feat.empty:
        st.error("Not enough data to build features.")
        st.stop()

    feat_cols = ['conso_moy', 'conso_std', 'max_hiver', 'max_ete',
                 'ratio_h_e', 'peak_morning', 'peak_evening', 'off_peak', 'ratio_we_wd']

    X_raw   = feat[feat_cols].values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_init  = 10
    pca_dim = 2

    # ── Elbow curve ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("How many groups? — The elbow method")
        st.markdown(
            "<p style='color:#8892aa;font-size:0.88rem;max-width:760px;margin-bottom:12px'>"
            "Each point on the blue curve is one K-Means run. "
            "<b style='color:#7ee8fa'>Inertia</b> (left axis) measures how tightly packed the clusters are — "
            "it always drops as k grows, but with diminishing returns. "
            "The <b>elbow</b> is where the curve stops dropping steeply: adding more clusters beyond "
            "that point barely helps. "
            "The <b style='color:#f4a261'>Silhouette score</b> (right axis, 0→1) independently confirms "
            "the best k: a higher score means clusters are well-separated and not overlapping. "
            "Pick the k where inertia bends <em>and</em> silhouette peaks."
            "</p>",
            unsafe_allow_html=True
        )

        inertias   = []
        sil_scores = []
        ks = range(2, 8)
        for ki in ks:
            km = KMeans(n_clusters=ki, random_state=42, n_init=n_init)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, km.labels_))

        # Best k by silhouette
        best_k     = list(ks)[int(np.argmax(sil_scores))]
        best_inert = inertias[best_k - 2]

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(
            x=list(ks), y=inertias, name="Inertia (tightness)",
            mode='lines+markers', line=dict(color=COLORS['primary'], width=2.5),
            marker=dict(size=8)
        ), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(
            x=list(ks), y=sil_scores, name="Silhouette score (separation)",
            mode='lines+markers', line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
            marker=dict(size=8)
        ), secondary_y=True)
        # Highlight the recommended k
        fig_elbow.add_vline(
            x=best_k, line_dash='dot', line_color=COLORS['green'], line_width=1.5,
        )
        fig_elbow.add_annotation(
            x=best_k, y=best_inert, xref='x', yref='y',
            text=f"  Best k = {best_k}",
            showarrow=False, font=dict(color=COLORS['green'], size=13),
            xanchor='left',
        )
        fig_elbow.update_layout(
            **PLOTLY_LAYOUT,
            title="Elbow curve & Silhouette score",
        )
        fig_elbow.update_yaxes(title_text="Inertia ↓ lower is tighter", secondary_y=False,
                               gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        fig_elbow.update_yaxes(title_text="Silhouette ↑ higher is better", secondary_y=True,
                               gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        fig_elbow.update_xaxes(title_text="Number of groups (k)", gridcolor=COLORS['grid'],
                               tickmode='linear', dtick=1)
        st.plotly_chart(fig_elbow, use_container_width=True)

        col_k, col_hint, _ = st.columns([1, 2, 1])
        with col_k:
            k = st.slider("Number of groups (k)", 2, 6, min(best_k, 6))
        with col_hint:
            st.markdown(
                f"<p style='color:#8892aa;font-size:0.82rem;margin-top:28px'>"
                f"Suggested by silhouette: <b style='color:#52d9a0'>k = {best_k}</b>. "
                f"You can override this with the slider.</p>",
                unsafe_allow_html=True
            )

    # ── K-Means final ────────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    labels   = km_final.fit_predict(X_scaled)
    feat['cluster'] = labels
    feat['cluster_label'] = feat['cluster'].map(
        {i: f"Class {i + 1}" for i in range(k)}
    )

    # Heuristic: the group with the highest weekend/weekday ratio → SR
    cluster_means = feat.groupby('cluster')[['ratio_we_wd', 'max_hiver', 'conso_moy']].mean()
    rs_cluster    = cluster_means['ratio_we_wd'].idxmax()
    feat['type']  = feat['cluster'].apply(lambda c: "SR (secondary)" if c == rs_cluster else "PR (primary)")

    palette = {f"Class {i + 1}": list(COLORS.values())[i] for i in range(k)}

    # ── PCA 2D ────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Meter map — who looks like whom?")
        st.caption(
            "Each dot is a meter. Dots that are close together have similar consumption patterns. "
            "Colors indicate the group detected automatically."
        )
        pca  = PCA(n_components=2)
        Xpca = pca.fit_transform(X_scaled)
        feat['PC1'] = Xpca[:, 0]
        feat['PC2'] = Xpca[:, 1]
        var1, var2 = pca.explained_variance_ratio_ * 100

        # Top contributing features per component (for axis label tooltip)
        loadings    = pd.DataFrame(pca.components_.T, index=feat_cols, columns=['PC1', 'PC2'])
        top_pc1     = loadings['PC1'].abs().nlargest(3).index.tolist()
        top_pc2     = loadings['PC2'].abs().nlargest(3).index.tolist()
        pc1_label   = f"PC1 ({var1:.1f}% variance) — mainly: {', '.join(top_pc1)}"
        pc2_label   = f"PC2 ({var2:.1f}% variance) — mainly: {', '.join(top_pc2)}"

        fig_pca = px.scatter(
            feat, x='PC1', y='PC2',
            color='cluster_label', symbol='type',
            hover_data=['id', 'conso_moy', 'ratio_h_e', 'ratio_we_wd'],
            color_discrete_map=palette,
            labels={'PC1': pc1_label, 'PC2': pc2_label, 'cluster_label': 'Group', 'type': 'Detected type'},
        )
        fig_pca.update_layout(**PLOTLY_LAYOUT, title="Meters grouped by consumption profile")
        fig_pca.update_traces(marker=dict(size=10, opacity=0.85))
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown(
            f"<p style='color:#8892aa;font-size:0.82rem;margin-top:-8px'>"
            f"The axes are <b>principal components</b> — linear combinations of the 9 features that "
            f"capture the most variance. "
            f"<b style='color:#7ee8fa'>PC1</b> ({var1:.1f}%) is driven mostly by "
            f"<code>{', '.join(top_pc1)}</code>. "
            f"<b style='color:#7ee8fa'>PC2</b> ({var2:.1f}%) is driven mostly by "
            f"<code>{', '.join(top_pc2)}</code>. "
            f"Together they explain <b>{var1+var2:.1f}%</b> of the total variance — "
            f"the closer two dots are, the more similar those meters' consumption profiles.</p>",
            unsafe_allow_html=True
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Group summary")
        summary = feat.groupby(['cluster_label', 'type']).agg(
            Count=('id', 'count'),
            Avg_consumption=('conso_moy', 'mean'),
            Winter_summer_ratio=('ratio_h_e', 'mean'),
            Weekend_weekday_ratio=('ratio_we_wd', 'mean'),
        ).round(2).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.session_state['feat_clustered'] = feat
    st.session_state['cluster_rs']     = rs_cluster


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Classification":
    st.markdown("<h1>Classification · SR vs PR</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892aa;max-width:720px'>"
        "Training and comparing two supervised models (Logistic Regression and MLP) "
        "to predict PR vs SR from consumption features. "
        "Labels are derived from K-Means clustering — see the <em>Methodology</em> "
        "expander inside each model tab for a full discussion of what this does and does not prove.</p>",
        unsafe_allow_html=True
    )

    # ── Data preparation ──────────────────────────────────────────────────────
    # Classification always uses k=2 (PR vs SR binary split), regardless of
    # whatever k was chosen in the Clustering tab.
    with st.spinner("Computing features & clustering (k=2)…"):
        feat = build_features(df_work, work_ids)
        feat_cols = ['conso_moy', 'conso_std', 'max_hiver', 'max_ete',
                     'ratio_h_e', 'peak_morning', 'peak_evening', 'off_peak', 'ratio_we_wd']
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feat[feat_cols].values)
        km       = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels   = km.fit_predict(X_scaled)
        feat['cluster'] = labels
        rs_cluster = feat.groupby('cluster')['ratio_we_wd'].mean().idxmax()
        feat['type'] = feat['cluster'].apply(lambda c: 1 if c == rs_cluster else 0)

    feat_cols = ['conso_moy', 'conso_std', 'max_hiver', 'max_ete',
                 'ratio_h_e', 'peak_morning', 'peak_evening', 'off_peak', 'ratio_we_wd']

    # Equilibrage
    rs_df = feat[feat['type'] == 1]
    rp_df = feat[feat['type'] == 0]
    n_min = min(len(rs_df), len(rp_df))

    if n_min < 4:
        st.warning("Not enough data to train a classifier (need at least 4 examples per class). Increase N_HOUSES.")
        st.stop()

    feat_bal = pd.concat([
        rs_df.sample(n_min, random_state=42),
        rp_df.sample(n_min, random_state=42),
    ])

    X   = feat_bal[feat_cols].values
    y   = feat_bal['type'].values
    sc2 = StandardScaler()
    X_s = sc2.fit_transform(X)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test set size (%)", 20, 40, 30) / 100
        with c2:
            seed = st.slider("Random seed", 0, 99, 42)

    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=test_size, random_state=seed, stratify=y)

    # ── Model training ────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0),
        "MLP (neural network)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        cm       = confusion_matrix(y_test, y_pred)
        report   = classification_report(y_test, y_pred, target_names=['PR', 'SR'], output_dict=True)
        results[name] = {'model': model, 'y_pred': y_pred, 'cm': cm, 'report': report}

    # ── Results ───────────────────────────────────────────────────────────────
    tabs = st.tabs(list(models.keys()))
    clr_list = [COLORS['primary'], COLORS['orange']]

    for i, (name, res) in enumerate(results.items()):
        with tabs[i]:
            rpt = res['report']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{rpt['accuracy']*100:.1f}%")
            c2.metric("Precision SR", f"{rpt['SR']['precision']*100:.1f}%")
            c3.metric("Recall SR",    f"{rpt['SR']['recall']*100:.1f}%")
            c4.metric("F1-score SR",  f"{rpt['SR']['f1-score']*100:.1f}%")

            col_a, col_b = st.columns(2)

            # Confusion matrix
            with col_a:
                cm   = res['cm']
                labs = ['RP', 'RS']
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    x=labs, y=labs,
                    color_continuous_scale=[[0, '#0b0e17'], [1, clr_list[i]]],
                    labels={'x': 'Predicted', 'y': 'Actual'},
                )
                fig_cm.update_layout(**PLOTLY_LAYOUT, title="Confusion matrix",
                                     xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                                     yaxis=dict(gridcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance (coefficients or first-layer weight norms)
            with col_b:
                model = res['model']
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                elif hasattr(model, 'coefs_'):
                    # MLP: norm of first-layer weights
                    importances = np.linalg.norm(model.coefs_[0], axis=1)
                else:
                    importances = np.zeros(len(feat_cols))

                df_imp = pd.DataFrame({'feature': feat_cols, 'importance': importances})
                df_imp = df_imp.sort_values('importance', ascending=True)

                fig_imp = px.bar(
                    df_imp, x='importance', y='feature', orientation='h',
                    color='importance', color_continuous_scale=['#1a2035', clr_list[i]],
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                )
                fig_imp.update_layout(**PLOTLY_LAYOUT, title="Feature importance",
                                      coloraxis_showscale=False,
                                      xaxis=_XAXIS,
                                      yaxis=dict(gridcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("Full classification report"):
                st.code(classification_report(y_test, res['y_pred'], target_names=['PR', 'SR']))

            with st.expander("ℹMethodology & limitations"):
                st.markdown("""
**What this classifier does**

The model learns to predict whether a meter is a *Primary Residence (PR)* or
*Secondary Residence (SR)* from 9 consumption features (averages, peaks, ratios).

**How the labels are generated**

Labels come directly from the K-Means clustering step: the cluster with the
highest weekend/weekday ratio is called SR, the rest PR.  
This means the classifier is **not** independently validated against ground-truth
labels — it is re-learning the clustering boundary in a supervised way.

**Why this is still useful**
- It tells you *which features matter most* for the SR/PR distinction (see feature importance).
- A trained classifier can label *new, unseen meters* in O(1) time without re-running K-Means.
- Comparing Logistic Regression vs MLP shows whether the boundary is linear or complex.

**Limitations to keep in mind**
- High accuracy here does **not** mean the clustering was correct — it only means
  the model successfully reproduced the clustering decisions.
- With only ~50 meters and a 70/30 split, the test set is very small (≈15 meters).
  Metrics can vary significantly across random seeds — try different seeds above.
- For a rigorous evaluation you would need ground-truth labels (e.g. from Enedis
  contract data stating primary vs secondary use).
                """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    st.markdown("<h1>Consumption Forecasting</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892aa;max-width:720px'>"
        "Predicting future daily consumption using ARIMA "
        "and a moving average baseline. Past readings and seasonal patterns "
        "are used as inputs.</p>",
        unsafe_allow_html=True
    )

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            sel_id = st.selectbox("Meter", work_ids)
        with c2:
            horizon = st.slider("Forecast horizon (days)", 7, 60, 14)

    df_h    = df_work[df_work['id'] == sel_id].copy()
    df_day  = df_h.groupby(df_h['horodate'].dt.date)['valeur'].mean().reset_index()
    df_day.columns = ['date', 'valeur']
    df_day  = df_day.sort_values('date').reset_index(drop=True)
    df_day['t'] = np.arange(len(df_day))

    if len(df_day) < 30:
        st.warning("Not enough days available for this meter.")
        st.stop()

    # ── Features for regression ───────────────────────────────────────────────
    for lag in [1, 7, 14]:
        df_day[f'lag_{lag}'] = df_day['valeur'].shift(lag)
    df_day['roll_7'] = df_day['valeur'].shift(1).rolling(7).mean()
    df_day['mois']   = pd.to_datetime(df_day['date']).dt.month
    df_day = df_day.dropna().reset_index(drop=True)

    feat_fc  = ['t', 'lag_1', 'lag_7', 'lag_14', 'roll_7', 'mois']
    X_fc     = df_day[feat_fc].values
    y_fc     = df_day['valeur'].values

    split = int(len(df_day) * 0.8)
    X_tr, X_te = X_fc[:split], X_fc[split:]
    y_tr, y_te = y_fc[:split], y_fc[split:]

    sc_fc  = StandardScaler()
    X_tr_s = sc_fc.fit_transform(X_tr)
    X_te_s = sc_fc.transform(X_te)

    lr     = LinearRegression()
    lr.fit(X_tr_s, y_tr)
    y_pred = lr.predict(X_te_s)

    # Bug fix 1 — recompute baseline independently from the raw 'valeur' column
    # so it cannot share information with the model's own 'roll_7' feature.
    y_base = (
        df_day['valeur']
        .shift(1).rolling(7).mean()
        .iloc[split:split + len(y_te)]
        .values
    )
    # Guard: if NaNs crept in (e.g. split < 7), fall back to the last known value
    if np.any(np.isnan(y_base)):
        y_base = np.where(np.isnan(y_base), df_day['valeur'].iloc[split - 1], y_base)

    mae_lr   = mean_absolute_error(y_te, y_pred)
    rmse_lr  = mean_squared_error(y_te, y_pred) ** 0.5
    r2_lr    = r2_score(y_te, y_pred)
    mae_base = mean_absolute_error(y_te, y_base)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (model)", f"{mae_lr:.1f} W")
    c2.metric("RMSE (model)", f"{rmse_lr:.1f} W")
    c3.metric("R² (model)", f"{r2_lr:.3f}")
    c4.metric("MAE (baseline)", f"{mae_base:.1f} W", delta=f"{mae_base-mae_lr:+.1f} W vs model", delta_color="inverse")

    # Bug fix 2 — use a safe slice instead of the fragile index-arithmetic comprehension
    test_dates = df_day['date'].iloc[split:split + len(y_te)].astype(str).tolist()

    with st.container(border=True):
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df_day['date'].astype(str).tolist(), y=df_day['valeur'],
            mode='lines', name='History',
            line=dict(color=COLORS['grid'], width=1.5)
        ))
        fig_fc.add_trace(go.Scatter(
            x=test_dates, y=y_te,
            mode='lines', name='Actual (test)',
            line=dict(color=COLORS['primary'], width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=test_dates, y=y_pred,
            mode='lines', name='Model prediction',
            line=dict(color=COLORS['orange'], width=2, dash='dash')
        ))
        fig_fc.add_trace(go.Scatter(
            x=test_dates, y=y_base,
            mode='lines', name='Baseline (moving avg.)',
            line=dict(color=COLORS['purple'], width=1.5, dash='dot')
        ))
        fig_fc.add_trace(go.Scatter(
            x=test_dates + test_dates[::-1],
            y=list(y_pred + rmse_lr) + list((y_pred - rmse_lr)[::-1]),
            fill='toself', fillcolor='rgba(244,162,97,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±RMSE interval', showlegend=True
        ))
        fig_fc.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Daily forecast — Meter {sel_id}",
            xaxis=dict(**_XAXIS, title="Date"),
            yaxis=dict(**_YAXIS, title="Avg. daily power (W)"),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with st.container(border=True):
        residus = y_te - y_pred
        fig_res = px.histogram(residus, nbins=30, color_discrete_sequence=[COLORS['green']],
                               labels={'value': 'Error (W)', 'count': 'Count'})
        fig_res.update_layout(**PLOTLY_LAYOUT, title="Prediction errors — how far off is the model?", xaxis=_XAXIS, yaxis=_YAXIS)
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Bug fix 3 — Future forecast using the horizon slider ──────────────────
    # The slider was previously defined but never used. We now generate a true
    # future forecast via recursive single-step prediction: each predicted value
    # is fed back as a lag input for the next step.
    with st.container(border=True):
        st.subheader(f"Future forecast — next {horizon} days")
        st.caption("Each future step is predicted from the previous ones (recursive forecasting).")

        current = df_day[['t', 'date', 'valeur', 'lag_1', 'lag_7', 'lag_14', 'roll_7', 'mois']].copy()
        future_dates  = []
        future_preds  = []

        hist_mean = float(df_day['valeur'].mean())   # anchor for damping
        hist_std  = float(df_day['valeur'].std())

        for i in range(1, horizon + 1):
            last      = current.iloc[-1]
            next_date = pd.to_datetime(last['date']) + pd.Timedelta(days=1)
            next_t    = int(last['t']) + 1
            # Always read lags from the original history where available,
            # falling back to predicted values only for the most recent steps.
            n_hist = len(df_day)
            def _lag(n):
                idx = -(n - i + 1)          # how far back in original history
                if i <= n:                  # still within original data
                    return float(df_day['valeur'].iloc[idx])
                return float(current['valeur'].iloc[-n]) if len(current) >= n else hist_mean
            lag_1    = _lag(1)
            lag_7    = _lag(7)
            lag_14   = _lag(14)
            roll_7_v = float(current['valeur'].iloc[-7:].mean()) if len(current) >= 7 else hist_mean
            mois_v   = int(next_date.month)

            x_new  = sc_fc.transform([[next_t, lag_1, lag_7, lag_14, roll_7_v, mois_v]])
            pred_v = float(lr.predict(x_new)[0])

            # Damp toward historical mean: prevent runaway collapse or explosion.
            # Weight shifts linearly from 0 (day 1) to 0.5 (last day).
            damp_w = 0.5 * (i / horizon)
            pred_v = (1 - damp_w) * pred_v + damp_w * hist_mean
            pred_v = max(pred_v, 0.0)   # consumption cannot be negative

            future_dates.append(str(next_date.date()))
            future_preds.append(pred_v)

            new_row = pd.DataFrame([{
                't': next_t, 'date': next_date.date(), 'valeur': pred_v,
                'lag_1': lag_1, 'lag_7': lag_7, 'lag_14': lag_14,
                'roll_7': roll_7_v, 'mois': mois_v,
            }])
            current = pd.concat([current, new_row], ignore_index=True)

        # Show last 30 days of history + future predictions
        n_context = min(30, len(df_day))
        ctx_dates = df_day['date'].iloc[-n_context:].astype(str).tolist()
        ctx_vals  = df_day['valeur'].iloc[-n_context:].tolist()

        fig_fut = go.Figure()
        fig_fut.add_trace(go.Scatter(
            x=ctx_dates, y=ctx_vals,
            mode='lines', name='Recent history',
            line=dict(color=COLORS['primary'], width=2),
        ))
        fig_fut.add_trace(go.Scatter(
            x=[ctx_dates[-1]] + future_dates,
            y=[ctx_vals[-1]]  + future_preds,
            mode='lines+markers', name=f'Forecast ({horizon} days)',
            line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
            marker=dict(size=6),
        ))
        fig_fut.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=[v + rmse_lr for v in future_preds] + [max(v - rmse_lr, 0) for v in future_preds[::-1]],
            fill='toself', fillcolor='rgba(244,162,97,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±RMSE confidence band',
        ))
        # Vertical separator between history and forecast
        # add_vline requires a numeric ms timestamp when x-axis uses string dates
        vline_x = pd.to_datetime(ctx_dates[-1]).timestamp() * 1000
        fig_fut.add_vline(
            x=vline_x,
            line_dash='dot', line_color=COLORS['green'],
            annotation_text='Today', annotation_position='top right',
        )
        fig_fut.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Future consumption forecast — Meter {sel_id}",
            xaxis=dict(**_XAXIS, title="Date"),
            yaxis=dict(**_YAXIS, title="Avg. daily power (W)", rangemode='tozero'),
        )
        st.plotly_chart(fig_fut, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — GÉNÉRATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Generation":
    st.markdown("<h1>Synthetic Profile Generation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892aa;max-width:720px'>"
        "Generate a realistic annual consumption profile conditioned on the residence type "
        "(SR or PR). Random noise and occasional spikes are added to mimic real-world variability. "
        "The result is compared against actual data to check coherence.</p>",
        unsafe_allow_html=True
    )

    if 'feat_clustered' not in st.session_state:
        with st.spinner("Computing features…"):
            feat = build_features(df_work, work_ids)
            feat_cols = ['conso_moy', 'conso_std', 'max_hiver', 'max_ete',
                         'ratio_h_e', 'peak_morning', 'peak_evening', 'off_peak', 'ratio_we_wd']
            sc = StandardScaler()
            Xs = sc.fit_transform(feat[feat_cols].values)
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            feat['cluster'] = km.fit_predict(Xs)
            rs_cluster = feat.groupby('cluster')['ratio_we_wd'].mean().idxmax()
            feat['type'] = feat['cluster'].apply(lambda c: "SR" if c == rs_cluster else "PR")
    else:
        feat = st.session_state['feat_clustered'].copy()
        rs_cluster = st.session_state['cluster_rs']
        feat['type'] = feat['cluster'].apply(lambda c: "SR" if c == rs_cluster else "PR")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            res_type  = st.selectbox("Residence type", ["PR (primary)", "SR (secondary)"])
        with c2:
            noise_pct = st.slider("Random noise (%)", 5, 40, 20)
        with c3:
            spike_pct = st.slider("Spike frequency (%)", 0, 20, 7)
        with c4:
            n_sample  = st.slider("Reference meters", 5, min(20, len(list_ids)), 10)

    type_key = "SR" if "SR" in res_type else "PR"
    ids_type = feat[feat['type'] == type_key]['id'].tolist()

    if len(ids_type) < 3:
        st.info("Not enough meters of this type. Run the Clustering step first with k=2.")
        st.stop()

    def generate_profile(ids_pool, df_src, noise, spike_freq, n_base):
        """Generate a synthetic weekly consumption profile."""
        sample_ids = np.random.choice(ids_pool, size=min(n_base, len(ids_pool)), replace=False)
        df_s       = df_src[df_src['id'].isin(sample_ids)].copy()
        df_g       = df_s.groupby('semaine_id')['valeur'].mean().reset_index()
        df_g       = df_g.sort_values('semaine_id').reset_index(drop=True)

        noise_arr  = np.random.normal(1, noise / 100, len(df_g))
        df_g['synth'] = df_g['valeur'] * noise_arr

        n_spikes = int(len(df_g) * spike_freq / 100)
        if n_spikes > 0:
            idx = np.random.choice(df_g.index, size=n_spikes, replace=False)
            df_g.loc[idx, 'synth'] += np.random.uniform(50, 150, size=n_spikes)

        df_g['synth'] = df_g['synth'].clip(lower=0)
        return df_g

    if st.button("Generate a new profile"):
        st.session_state.gen_seed = np.random.randint(0, 10_000)

    np.random.seed(st.session_state.get('gen_seed', 0))
    df_synth = generate_profile(ids_type, df_work, noise_pct, spike_pct, n_sample)

    df_real_avg = (
        df_work[df_work['id'].isin(ids_type)]
        .groupby('semaine_id')['valeur'].mean().reset_index()
        .sort_values('semaine_id')
    )

    with st.container(border=True):
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Scatter(
            x=df_real_avg['semaine_id'], y=df_real_avg['valeur'],
            mode='lines', name=f'Real average ({type_key})',
            line=dict(color=COLORS['primary'], width=2, dash='dot'),
        ))
        fig_gen.add_trace(go.Scatter(
            x=df_synth['semaine_id'], y=df_synth['synth'],
            mode='lines+markers', name=f'Synthetic profile ({type_key})',
            line=dict(color=COLORS['orange'], width=2.5),
            marker=dict(size=5),
        ))
        fig_gen.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Synthetic vs real profile — {res_type}",
            xaxis=dict(
                tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS,
                title='Month', gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']
            ),
            yaxis=dict(range=[0, None], title='Power (W)', gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    df_merged = pd.merge(
        df_synth[['semaine_id', 'synth']],
        df_real_avg[['semaine_id', 'valeur']],
        on='semaine_id', how='inner'
    )

    if len(df_merged) >= 2:
        mae_g  = mean_absolute_error(df_merged['valeur'], df_merged['synth'])
        rmse_g = mean_squared_error(df_merged['valeur'], df_merged['synth']) ** 0.5
        r2_g   = r2_score(df_merged['valeur'], df_merged['synth'])

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE vs real", f"{mae_g:.1f} W")
        c2.metric("RMSE vs real", f"{rmse_g:.1f} W")
        c3.metric("R² similarity", f"{r2_g:.3f}")

    with st.container(border=True):
        st.subheader("PR vs SR — average real profiles side by side")
        rp_ids = feat[feat['type'] == 'PR']['id'].tolist()
        rs_ids = feat[feat['type'] == 'SR']['id'].tolist()

        rp_avg = df_work[df_work['id'].isin(rp_ids)].groupby('semaine_id')['valeur'].mean().reset_index()
        rs_avg = df_work[df_work['id'].isin(rs_ids)].groupby('semaine_id')['valeur'].mean().reset_index()

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=rp_avg['semaine_id'], y=rp_avg['valeur'],
            mode='lines', name='PR average',
            line=dict(color=COLORS['primary'], width=2.5),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=rs_avg['semaine_id'], y=rs_avg['valeur'],
            mode='lines', name='SR average',
            line=dict(color=COLORS['orange'], width=2.5, dash='dash'),
        ))
        fig_cmp.update_layout(
            **PLOTLY_LAYOUT,
            title="Weekly average profiles — PR vs SR",
            xaxis=dict(
                tickmode='array', tickvals=MOIS_TICKS, ticktext=MOIS_LABELS,
                gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']
            ),
            yaxis=dict(range=[0, None], title='Power (W)', gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.caption(
        "**Key insight**: a secondary residence (SR) typically shows higher weekend consumption "
        "and holiday spikes, while a primary residence (PR) has steady daily usage with a strong "
        "winter peak from electric heating (RES2)."
    )