"""
backend.py — Chargement des données, feature engineering, modèles ML.
Aucun import Streamlit ici : uniquement du calcul pur.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    silhouette_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

# ── Constantes globales ───────────────────────────────────────────────────────
N_INITIAL = 100
N_SAMPLE  = 50

FEAT_COLS = [
    'conso_moy', 'conso_std', 'max_hiver', 'max_ete',
    'ratio_h_e', 'peak_morning', 'peak_evening', 'off_peak', 'ratio_we_wd'
]

MOIS_LABELS = ['Nov', 'Dec', 'Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct']
MOIS_TICKS  = [i * (52 / 12) for i in range(12)]


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_preview(path, skip=0):
    try:
        if skip == 0:
            return pd.read_csv(path, nrows=5)
        return pd.read_csv(path, skiprows=range(1, skip + 1), nrows=5)
    except Exception:
        return pd.read_csv(path, nrows=5)


def load_houses(path, n=None):
    """Charge les n premiers compteurs depuis le CSV long-format."""
    reader    = pd.read_csv(path, chunksize=100_000)
    data_list = []
    found_ids = []

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        for idx in chunk['id'].unique():
            if idx not in found_ids:
                found_ids.append(idx)
        if n is not None and len(found_ids) >= n:
            data_list.append(chunk[chunk['id'].isin(found_ids[:n])])
            break
        data_list.append(chunk)

    df = pd.concat(data_list, ignore_index=True)
    if n is not None:
        found_ids = found_ids[:n]
        df = df[df['id'].isin(found_ids)]

    df['horodate']    = pd.to_datetime(df['horodate'], utc=True, errors='coerce')
    df = df.dropna(subset=['horodate'])
    df['mois']        = df['horodate'].dt.month
    df['jour']        = df['horodate'].dt.day
    df['heure']       = df['horodate'].dt.hour
    df['dow']         = df['horodate'].dt.dayofweek
    df['mois_fiscal'] = (df['mois'] - 11) % 12
    df['semaine_id']  = (df['mois_fiscal'] * 4 + (df['jour'] - 1) // 7).clip(0, 51)

    return df, found_ids


def get_work_sample(df_all, list_ids, sample_n=N_SAMPLE, seed=42):
    rng     = np.random.default_rng(seed)
    w_ids   = list(rng.choice(list_ids, size=min(sample_n, len(list_ids)), replace=False))
    df_work = df_all[df_all['id'].isin(w_ids)].copy()
    return df_work, w_ids


# ══════════════════════════════════════════════════════════════════════════════
#  EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def get_exploration_kpis(df_all, list_ids, work_ids):
    return {
        'n_meters'  : len(list_ids),
        'n_readings': len(df_all),
        'avg_power' : df_all['valeur'].mean(),
        'total_kwh' : df_all['valeur'].sum() * 0.5 / 1000,
        'n_work'    : len(work_ids),
    }


def get_yearly_avg(df_all):
    return (
        df_all.groupby('semaine_id')['valeur']
        .mean().reset_index().sort_values('semaine_id')
    )


def get_meter_weekly(df_all, meter_id):
    return (
        df_all[df_all['id'] == meter_id]
        .groupby('semaine_id')['valeur'].mean().reset_index()
    )


def get_heatmap_data(df_all):
    df_heat = df_all.groupby(['dow', 'heure'])['valeur'].mean().reset_index()
    return (
        df_heat.pivot(index='heure', columns='dow', values='valeur')
        .reindex(index=range(24), columns=range(7))
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df_all, ids):
    rows = []
    for hid in ids:
        d = df_all[df_all['id'] == hid]
        if len(d) < 50:
            continue

        conso_moy = d['valeur'].mean()
        conso_std = d['valeur'].std()
        hiver     = d[d['mois'].isin([11, 12, 1, 2, 3])]['valeur'].mean()
        ete       = d[d['mois'].isin([6, 7, 8])]['valeur'].mean()
        ratio_h_e = hiver / ete if ete > 0 else np.nan

        peak_morn = d[d['heure'].between(6, 9)]['valeur'].mean()
        peak_even = d[d['heure'].between(18, 22)]['valeur'].mean()
        off_peak  = d[d['heure'].between(0, 5)]['valeur'].mean()

        dow_s       = d['horodate'].dt.dayofweek
        we          = d[dow_s >= 5]['valeur'].mean()
        wd          = d[dow_s < 5]['valeur'].mean()
        ratio_we_wd = we / wd if wd > 0 else np.nan

        rows.append({
            'id'          : hid,
            'conso_moy'   : conso_moy,
            'conso_std'   : conso_std,
            'max_hiver'   : hiver,
            'max_ete'     : ete,
            'ratio_h_e'   : ratio_h_e,
            'peak_morning': peak_morn,
            'peak_evening': peak_even,
            'off_peak'    : off_peak,
            'ratio_we_wd' : ratio_we_wd,
        })

    return pd.DataFrame(rows).dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def run_elbow(X_scaled, k_range=range(2, 8), n_init=10):
    inertias, sil_scores = [], []
    for ki in k_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=n_init)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    return list(k_range), inertias, sil_scores, best_k


def run_clustering(feat, k, n_init=10):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat[FEAT_COLS].values)

    km_final      = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    labels        = km_final.fit_predict(X_scaled)
    feat          = feat.copy()
    feat['cluster']       = labels
    feat['cluster_label'] = feat['cluster'].map({i: f"Groupe {i+1}" for i in range(k)})

    cluster_means = feat.groupby('cluster')[['ratio_we_wd', 'max_hiver', 'conso_moy']].mean()
    rs_cluster    = cluster_means['ratio_we_wd'].idxmax()
    feat['type']  = feat['cluster'].apply(lambda c: "RS (secondaire)" if c == rs_cluster else "RP (principale)")

    pca      = PCA(n_components=2)
    Xpca     = pca.fit_transform(X_scaled)
    feat['PC1'] = Xpca[:, 0]
    feat['PC2'] = Xpca[:, 1]

    var1, var2 = pca.explained_variance_ratio_ * 100
    loadings   = pd.DataFrame(pca.components_.T, index=FEAT_COLS, columns=['PC1', 'PC2'])
    top_pc1    = loadings['PC1'].abs().nlargest(3).index.tolist()
    top_pc2    = loadings['PC2'].abs().nlargest(3).index.tolist()

    pca_info = {'var1': var1, 'var2': var2, 'top_pc1': top_pc1, 'top_pc2': top_pc2}

    summary = feat.groupby(['cluster_label', 'type']).agg(
        Effectif=('id', 'count'),
        Conso_moyenne=('conso_moy', 'mean'),
        Ratio_hiver_ete=('ratio_h_e', 'mean'),
        Ratio_we_semaine=('ratio_we_wd', 'mean'),
    ).round(2).reset_index()

    return feat, rs_cluster, pca_info, summary


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_classification_data(df_work, work_ids, test_size=0.30, seed=42):
    feat     = build_features(df_work, work_ids)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat[FEAT_COLS].values)
    km       = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels   = km.fit_predict(X_scaled)
    feat['cluster'] = labels
    rs_cluster      = feat.groupby('cluster')['ratio_we_wd'].mean().idxmax()
    feat['type']    = feat['cluster'].apply(lambda c: 1 if c == rs_cluster else 0)

    rs_df = feat[feat['type'] == 1]
    rp_df = feat[feat['type'] == 0]
    n_min = min(len(rs_df), len(rp_df))

    if n_min < 4:
        return None, None, None, None, n_min

    feat_bal = pd.concat([
        rs_df.sample(n_min, random_state=42),
        rp_df.sample(n_min, random_state=42),
    ])

    X   = feat_bal[FEAT_COLS].values
    y   = feat_bal['type'].values
    sc2 = StandardScaler()
    X_s = sc2.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, n_min


def train_classifiers(X_train, X_test, y_train, y_test):
    models = {
        "Regression Logistique": LogisticRegression(max_iter=500, C=1.0),
        "MLP (reseau de neurones)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred     = model.predict(X_test)
        cm         = confusion_matrix(y_test, y_pred)
        report     = classification_report(y_test, y_pred, target_names=['RP', 'RS'], output_dict=True)
        report_txt = classification_report(y_test, y_pred, target_names=['RP', 'RS'])

        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'coefs_'):
            importances = np.linalg.norm(model.coefs_[0], axis=1)
        else:
            importances = np.zeros(len(FEAT_COLS))

        df_imp = (
            pd.DataFrame({'feature': FEAT_COLS, 'importance': importances})
            .sort_values('importance', ascending=True)
        )

        results[name] = {
            'model'      : model,
            'y_pred'     : y_pred,
            'cm'         : cm,
            'report'     : report,
            'report_txt' : report_txt,
            'df_imp'     : df_imp,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def run_forecasting(df_work, meter_id, horizon=14):
    df_h   = df_work[df_work['id'] == meter_id].copy()
    df_day = df_h.groupby(df_h['horodate'].dt.date)['valeur'].mean().reset_index()
    df_day.columns = ['date', 'valeur']
    df_day = df_day.sort_values('date').reset_index(drop=True)
    df_day['t'] = np.arange(len(df_day))

    if len(df_day) < 30:
        return None

    for lag in [1, 7, 14]:
        df_day[f'lag_{lag}'] = df_day['valeur'].shift(lag)
    df_day['roll_7'] = df_day['valeur'].shift(1).rolling(7).mean()
    df_day['mois']   = pd.to_datetime(df_day['date']).dt.month
    df_day = df_day.dropna().reset_index(drop=True)

    feat_fc = ['t', 'lag_1', 'lag_7', 'lag_14', 'roll_7', 'mois']
    X_fc    = df_day[feat_fc].values
    y_fc    = df_day['valeur'].values

    split  = int(len(df_day) * 0.8)
    X_tr, X_te = X_fc[:split], X_fc[split:]
    y_tr, y_te = y_fc[:split], y_fc[split:]

    sc_fc  = StandardScaler()
    X_tr_s = sc_fc.fit_transform(X_tr)
    X_te_s = sc_fc.transform(X_te)

    lr     = LinearRegression()
    lr.fit(X_tr_s, y_tr)
    y_pred = lr.predict(X_te_s)

    y_base = (
        df_day['valeur'].shift(1).rolling(7).mean()
        .iloc[split:split + len(y_te)].values
    )
    if np.any(np.isnan(y_base)):
        y_base = np.where(np.isnan(y_base), df_day['valeur'].iloc[split - 1], y_base)

    mae_lr   = mean_absolute_error(y_te, y_pred)
    rmse_lr  = mean_squared_error(y_te, y_pred) ** 0.5
    r2_lr    = r2_score(y_te, y_pred)
    mae_base = mean_absolute_error(y_te, y_base)
    test_dates = df_day['date'].iloc[split:split + len(y_te)].astype(str).tolist()

    # Forecast recursif
    current   = df_day[['t', 'date', 'valeur', 'lag_1', 'lag_7', 'lag_14', 'roll_7', 'mois']].copy()
    hist_mean = float(df_day['valeur'].mean())
    future_dates, future_preds = [], []

    for i in range(1, horizon + 1):
        last      = current.iloc[-1]
        next_date = pd.to_datetime(last['date']) + pd.Timedelta(days=1)
        next_t    = int(last['t']) + 1

        def _lag(n):
            if i <= n:
                return float(df_day['valeur'].iloc[-(n - i + 1)])
            return float(current['valeur'].iloc[-n]) if len(current) >= n else hist_mean

        lag_1    = _lag(1)
        lag_7    = _lag(7)
        lag_14   = _lag(14)
        roll_7_v = float(current['valeur'].iloc[-7:].mean()) if len(current) >= 7 else hist_mean
        mois_v   = int(next_date.month)

        x_new  = sc_fc.transform([[next_t, lag_1, lag_7, lag_14, roll_7_v, mois_v]])
        pred_v = float(lr.predict(x_new)[0])
        damp_w = 0.5 * (i / horizon)
        pred_v = max((1 - damp_w) * pred_v + damp_w * hist_mean, 0.0)

        future_dates.append(str(next_date.date()))
        future_preds.append(pred_v)

        new_row = pd.DataFrame([{
            't': next_t, 'date': next_date.date(), 'valeur': pred_v,
            'lag_1': lag_1, 'lag_7': lag_7, 'lag_14': lag_14,
            'roll_7': roll_7_v, 'mois': mois_v,
        }])
        current = pd.concat([current, new_row], ignore_index=True)

    n_ctx     = min(30, len(df_day))
    ctx_dates = df_day['date'].iloc[-n_ctx:].astype(str).tolist()
    ctx_vals  = df_day['valeur'].iloc[-n_ctx:].tolist()

    return {
        'df_day'      : df_day,
        'test_dates'  : test_dates,
        'y_te'        : y_te,
        'y_pred'      : y_pred,
        'y_base'      : y_base,
        'mae_lr'      : mae_lr,
        'rmse_lr'     : rmse_lr,
        'r2_lr'       : r2_lr,
        'mae_base'    : mae_base,
        'future_dates': future_dates,
        'future_preds': future_preds,
        'ctx_dates'   : ctx_dates,
        'ctx_vals'    : ctx_vals,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def get_typed_ids(df_work, work_ids):
    feat     = build_features(df_work, work_ids)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat[FEAT_COLS].values)
    km       = KMeans(n_clusters=2, random_state=42, n_init=10)
    feat['cluster'] = km.fit_predict(X_scaled)
    rs_cluster      = feat.groupby('cluster')['ratio_we_wd'].mean().idxmax()
    feat['type']    = feat['cluster'].apply(lambda c: "RS" if c == rs_cluster else "RP")
    return feat, rs_cluster


def generate_profile(ids_pool, df_src, noise_pct, spike_pct, n_base, seed=0):
    np.random.seed(seed)
    sample_ids    = np.random.choice(ids_pool, size=min(n_base, len(ids_pool)), replace=False)
    df_s          = df_src[df_src['id'].isin(sample_ids)].copy()
    df_g          = df_s.groupby('semaine_id')['valeur'].mean().reset_index()
    df_g          = df_g.sort_values('semaine_id').reset_index(drop=True)
    noise_arr     = np.random.normal(1, noise_pct / 100, len(df_g))
    df_g['synth'] = df_g['valeur'] * noise_arr
    n_spikes      = int(len(df_g) * spike_pct / 100)
    if n_spikes > 0:
        idx = np.random.choice(df_g.index, size=n_spikes, replace=False)
        df_g.loc[idx, 'synth'] += np.random.uniform(50, 150, size=n_spikes)
    df_g['synth'] = df_g['synth'].clip(lower=0)
    return df_g


def get_real_avg_by_type(df_work, feat_typed, type_key):
    ids = feat_typed[feat_typed['type'] == type_key]['id'].tolist()
    return (
        df_work[df_work['id'].isin(ids)]
        .groupby('semaine_id')['valeur'].mean().reset_index()
        .sort_values('semaine_id')
    ), ids


def score_generation(df_synth, df_real_avg):
    df_m = pd.merge(
        df_synth[['semaine_id', 'synth']],
        df_real_avg[['semaine_id', 'valeur']],
        on='semaine_id', how='inner'
    )
    if len(df_m) < 2:
        return None
    return {
        'mae' : mean_absolute_error(df_m['valeur'], df_m['synth']),
        'rmse': mean_squared_error(df_m['valeur'], df_m['synth']) ** 0.5,
        'r2'  : r2_score(df_m['valeur'], df_m['synth']),
    }
