"""RouteWise AI core: synthetic shipment generator, validation, delay model, evaluation and scoring.

Prediction moment: **dispatch**. Model inputs are limited to information known when a shipment
is dispatched (booking details, forecasts, carrier history). Anything that happens in transit is
an *outcome* and is never used as an input.

Evaluation protocol:
1. Records are ordered by dispatch date; the latest 25% form a forward-in-time holdout.
2. Candidate models are compared by 5-fold stratified CV PR-AUC on the earlier 75% only.
3. The decision threshold is tuned on out-of-fold *training* predictions (max F1).
4. Dashboard scores are never in-sample: training-period rows get out-of-fold predictions and
   holdout rows are scored by the model fitted on the training period.
5. The model used for new predictions is refit on all rows.

All data is synthetic. Metrics describe how well models recover the simulated mechanism and say
nothing about real-world delivery performance.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- schema
NUMERIC = ['distance_km', 'planned_days', 'weight_kg', 'weather_severity', 'traffic_index',
           'warehouse_backlog', 'carrier_on_time_rate', 'customs_clearance_required']
CATEGORICAL = ['transport_mode', 'region', 'priority']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'delayed'
# Known only after dispatch: never model inputs
OUTCOMES = ['delayed', 'delay_days', 'actual_transit_days', 'customs_hold_hours']
ID_COLUMNS = ['shipment_id', 'dispatch_date']

MODES = ['Road', 'Rail', 'Air', 'Sea']
REGIONS = ['North', 'South', 'East', 'West', 'Central']
PRIORITIES = ['Standard', 'Express', 'Critical']
CATEGORY_VALUES = {'transport_mode': MODES, 'region': REGIONS, 'priority': PRIORITIES}
RANGES = {  # inclusive validation ranges for inputs
    'distance_km': (1, 20000), 'planned_days': (1, 60), 'weight_kg': (0.1, 50000),
    'weather_severity': (0, 5), 'traffic_index': (0, 100), 'warehouse_backlog': (0, 100),
    'carrier_on_time_rate': (0, 1), 'customs_clearance_required': (0, 1),
}
FEATURE_INFO = {
    'distance_km': 'Route distance (booking)',
    'planned_days': 'Planned transit days (booking)',
    'weight_kg': 'Shipment weight (booking)',
    'weather_severity': 'Weather forecast severity 0–5 (at dispatch)',
    'traffic_index': 'Traffic/congestion forecast 0–100 (at dispatch)',
    'warehouse_backlog': 'Origin warehouse backlog 0–100 (at dispatch)',
    'carrier_on_time_rate': "Carrier's historical on-time rate (before dispatch)",
    'customs_clearance_required': 'Customs clearance required, 0/1 (booking)',
    'transport_mode': 'Road / Rail / Air / Sea (booking)',
    'region': 'Destination region (booking)',
    'priority': 'Standard / Express / Critical (booking)',
}
RISK_BANDS = (0.20, 0.45)   # Low < 0.20 <= Medium < 0.45 <= High (illustrative, see docs)
RANDOM_STATE = 42
HOLDOUT_SHARE = 0.25


# ---------------------------------------------------------------- synthetic data
def generate_data(n: int = 6000, seed: int = 42, missing_rate: float = 0.015) -> pd.DataFrame:
    """Seeded synthetic shipments dispatched during 2025 with documented, simulated mechanics."""
    if n < 100:
        raise ValueError('n must be >= 100')
    rng = np.random.default_rng(seed)
    start = np.datetime64('2025-01-01')
    dispatch = np.sort(start + rng.integers(0, 365, n).astype('timedelta64[D]'))
    month = pd.DatetimeIndex(dispatch).month.to_numpy()
    winter = np.isin(month, [12, 1, 2])
    peak = np.isin(month, [11, 12])

    mode = rng.choice(MODES, n, p=[.58, .17, .13, .12])
    region = rng.choice(REGIONS, n)
    priority = rng.choice(PRIORITIES, n, p=[.68, .25, .07])
    distance = np.round(rng.uniform(20, 3000, n), 1)
    speed = np.where(mode == 'Air', 1250, np.where(mode == 'Rail', 550, np.where(mode == 'Sea', 350, 400)))
    planned = np.maximum(1, np.ceil(distance / speed)).astype(int)
    weight = np.round(rng.lognormal(5.0, 1.0, n), 1)
    # forecasts known at dispatch (seasonal)
    weather = np.where(winter, rng.choice(6, n, p=[.08, .17, .25, .25, .15, .10]),
                       rng.choice(6, n, p=[.25, .30, .22, .13, .07, .03]))
    traffic = rng.integers(10, 101, n).astype(float)
    backlog = np.clip(rng.integers(0, 81, n) + np.where(peak, rng.integers(0, 21, n), 0), 0, 100)
    ontime = np.round(rng.uniform(.65, .99, n), 3)
    customs_req = (rng.random(n) < np.select([mode == 'Sea', mode == 'Air', mode == 'Rail'], [.60, .45, .25], .10)).astype(int)

    # in-transit events (unknown at dispatch)
    hold = np.where(customs_req == 1, rng.choice([0, 0, 4, 8, 16, 24, 36], n), 0)
    weather_realised = np.clip(weather + rng.integers(-1, 2, n), 0, 5)

    logit = (-5.3 + .55 * weather_realised + .026 * traffic + .022 * backlog - 6.0 * (ontime - .75)
             + .055 * hold + .00025 * distance + np.where(mode == 'Sea', .35, 0)
             - np.where(priority == 'Critical', .5, 0) - np.where(priority == 'Express', .2, 0))
    p = 1 / (1 + np.exp(-logit))
    delayed = rng.binomial(1, p)
    delay_days = np.where(delayed == 1, 1 + rng.poisson(.6 + hold / 24 + .15 * weather_realised), 0)

    df = pd.DataFrame({
        'shipment_id': [f'RW-{i:06d}' for i in range(1, n + 1)],
        'dispatch_date': pd.to_datetime(dispatch).date,
        'distance_km': distance, 'planned_days': planned, 'weight_kg': weight,
        'weather_severity': weather, 'traffic_index': traffic, 'warehouse_backlog': backlog,
        'carrier_on_time_rate': ontime, 'customs_clearance_required': customs_req,
        'transport_mode': mode, 'region': region, 'priority': priority,
        'delayed': delayed, 'delay_days': delay_days,
        'actual_transit_days': planned + delay_days, 'customs_hold_hours': hold,
    })
    if missing_rate > 0:   # realistic gaps in upstream feeds; handled by imputation
        for col, rate in [('traffic_index', missing_rate), ('carrier_on_time_rate', missing_rate / 2)]:
            df.loc[rng.random(n) < rate, col] = np.nan
    return df


# ---------------------------------------------------------------- validation
@dataclass
class QualityReport:
    rows: int
    missing: dict
    dropped_rows: int
    notes: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{'column': c, 'missing values': v, 'share': v / max(self.rows, 1)}
                             for c, v in self.missing.items()])


def validate_shipments(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    """Schema, category and range checks. Missing numeric values are kept (imputed in the model);
    rows with invalid categories or out-of-range values are dropped and counted."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Expected a DataFrame of shipments')
    need = FEATURES + ([TARGET] if require_target else [])
    missing_cols = sorted(set(need) - set(df.columns))
    if missing_cols:
        raise ValueError(f'Missing columns: {missing_cols}')
    data = df.copy()
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    keep = pd.Series(True, index=data.index)
    for col, allowed in CATEGORY_VALUES.items():
        data[col] = data[col].astype(str).str.strip().str.title()
        keep &= data[col].isin(allowed)
    for col, (lo, hi) in RANGES.items():
        keep &= data[col].isna() | data[col].between(lo, hi)
    if require_target:
        keep &= data[TARGET].isin([0, 1])
    report = QualityReport(len(data), {c: int(data[c].isna().sum()) for c in NUMERIC}, int((~keep).sum()))
    data = data[keep].copy()
    if require_target:
        data[TARGET] = data[TARGET].astype(int)
    if data.empty:
        raise ValueError('No valid shipments after validation')
    if report.dropped_rows:
        report.notes.append(f'{report.dropped_rows} rows dropped (invalid category or out-of-range value).')
    return data, report


def validate_input(record: dict) -> pd.DataFrame:
    """Validate a single prediction request; raises ValueError listing every problem."""
    problems = []
    for col in FEATURES:
        if col not in record or record[col] is None or (isinstance(record[col], float) and np.isnan(record[col])):
            problems.append(f'{col} is required')
    for col, (lo, hi) in RANGES.items():
        v = record.get(col)
        if v is not None and not problems:
            try:
                if not lo <= float(v) <= hi:
                    problems.append(f'{col} must be between {lo} and {hi}')
            except (TypeError, ValueError):
                problems.append(f'{col} must be numeric')
    for col, allowed in CATEGORY_VALUES.items():
        if record.get(col) not in allowed:
            problems.append(f'{col} must be one of {allowed}')
    if problems:
        raise ValueError('; '.join(problems))
    return pd.DataFrame([{c: record[c] for c in FEATURES}])


# ---------------------------------------------------------------- models
def _preprocessor(scale: bool, numeric: list[str] | None = None) -> ColumnTransformer:
    num_steps = [('impute', SimpleImputer(strategy='median'))] + ([('scale', StandardScaler())] if scale else [])
    return ColumnTransformer([
        ('num', Pipeline(num_steps), numeric or NUMERIC),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])


def make_model(name: str, numeric: list[str] | None = None) -> Pipeline:
    """Build one of the candidate pipelines (optionally on a different numeric column list)."""
    if name == 'Logistic Regression':
        return Pipeline([('preprocess', _preprocessor(True, numeric)), ('model', LogisticRegression(max_iter=2000, C=1.0))])
    balanced = name == 'Random Forest (balanced)'
    return Pipeline([('preprocess', _preprocessor(False, numeric)), ('model', RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight='balanced' if balanced else None,
        random_state=RANDOM_STATE, n_jobs=-1))])


MODEL_NAMES = ['Random Forest (balanced)', 'Random Forest (unweighted)', 'Logistic Regression']


def candidate_models() -> dict[str, Pipeline]:
    return {name: make_model(name) for name in MODEL_NAMES}


def build_pipeline() -> Pipeline:
    """Original RouteWise Random Forest (class-balanced)."""
    return make_model('Random Forest (balanced)')


def temporal_split(df: pd.DataFrame, share: float = HOLDOUT_SHARE):
    """Latest `share` of shipments (by dispatch date, then id) is the holdout."""
    order = df.sort_values(['dispatch_date', 'shipment_id'] if 'shipment_id' in df else ['dispatch_date']).index \
        if 'dispatch_date' in df.columns else df.index
    n_test = int(round(len(df) * share))
    return df.loc[order[:-n_test]], df.loc[order[-n_test:]]


def _threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(thr), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def _best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


@dataclass
class Evaluation:
    model: Pipeline                # selected model refit on all rows (new predictions)
    holdout_model: Pipeline        # selected model fitted on the training period only
    model_name: str
    comparison: pd.DataFrame
    metrics: dict
    tuned: dict
    random_split: dict
    leakage_check: dict
    curves: dict
    importance: pd.DataFrame
    scores: pd.Series              # never-in-sample probability for every row
    holdout_index: pd.Index
    meta: dict


def evaluate(df: pd.DataFrame, n_folds: int = 5) -> Evaluation:
    data, _ = validate_shipments(df)
    if data[TARGET].nunique() < 2 or data[TARGET].value_counts().min() < 10:
        raise ValueError('Need at least 10 delayed and 10 on-time shipments')
    train, test = temporal_split(data)
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

    rows, oof = [], {}
    for name, pipe in candidate_models().items():
        p = cross_val_predict(pipe, X_tr, y_tr, cv=folds, method='predict_proba')[:, 1]
        oof[name] = p
        rows.append({'Model': name, 'CV PR-AUC': average_precision_score(y_tr, p), 'CV ROC-AUC': roc_auc_score(y_tr, p),
                     'CV Brier': brier_score_loss(y_tr, p), 'CV recall @0.5': recall_score(y_tr, p >= .5),
                     'CV precision @0.5': precision_score(y_tr, p >= .5, zero_division=0)})
    comparison = pd.DataFrame(rows).sort_values('CV PR-AUC', ascending=False).reset_index(drop=True)
    best = comparison.iloc[0]['Model']
    thr = _best_f1_threshold(y_tr, oof[best])

    holdout_model = candidate_models()[best].fit(X_tr, y_tr)
    p_te = holdout_model.predict_proba(X_te)[:, 1]
    metrics = {'roc_auc': float(roc_auc_score(y_te, p_te)), 'average_precision': float(average_precision_score(y_te, p_te)),
               'brier': float(brier_score_loss(y_te, p_te)), 'positive_rate': float(y_te.mean()),
               'train_rows': int(len(X_tr)), 'test_rows': int(len(X_te)),
               'report': classification_report(y_te, (p_te >= .5).astype(int), output_dict=True, zero_division=0),
               'confusion_matrix': confusion_matrix(y_te, (p_te >= .5).astype(int), labels=[0, 1]).tolist(),
               **{f'{k}@0.5': v for k, v in _threshold_metrics(y_te, p_te, .5).items() if k != 'threshold'}}
    tuned = _threshold_metrics(y_te, p_te, thr)

    # comparison: a random (non-temporal) split of the same data
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(data[FEATURES], data[TARGET], test_size=HOLDOUT_SHARE,
                                                  stratify=data[TARGET], random_state=RANDOM_STATE)
    pr = candidate_models()[best].fit(Xr_tr, yr_tr).predict_proba(Xr_te)[:, 1]
    random_split = {'roc_auc': float(roc_auc_score(yr_te, pr)), 'average_precision': float(average_precision_score(yr_te, pr))}

    # leakage demonstration: what happens if the post-dispatch customs hold were (wrongly) used
    leakage_check = {}
    if 'customs_hold_hours' in data.columns:
        leaky_feats = FEATURES + ['customs_hold_hours']
        leaky = make_model(best, NUMERIC + ['customs_hold_hours'])
        pl = leaky.fit(train[leaky_feats], y_tr).predict_proba(test[leaky_feats])[:, 1]
        leakage_check = {'leaky_roc_auc': float(roc_auc_score(y_te, pl)),
                         'leaky_average_precision': float(average_precision_score(y_te, pl))}

    prec, rec, _ = precision_recall_curve(y_te, p_te)
    fpr, tpr, _ = roc_curve(y_te, p_te)
    frac, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy='quantile')
    curves = {'pr': pd.DataFrame({'Recall': rec, 'Precision': prec}),
              'roc': pd.DataFrame({'False positive rate': fpr, 'True positive rate': tpr}),
              'calibration': pd.DataFrame({'Mean predicted probability': mean_pred, 'Observed delay rate': frac})}
    imp = permutation_importance(holdout_model, X_te, y_te, scoring='average_precision', n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({'Feature': FEATURES, 'PR-AUC drop': imp.importances_mean, 'Std': imp.importances_std}
                              ).sort_values('PR-AUC drop', ascending=False).reset_index(drop=True)

    scores = pd.concat([pd.Series(oof[best], index=X_tr.index), pd.Series(p_te, index=X_te.index)]).reindex(data.index)
    final = candidate_models()[best].fit(data[FEATURES], data[TARGET])
    meta = {'threshold': thr, 'n_folds': n_folds, 'random_state': RANDOM_STATE, 'sklearn': sklearn.__version__,
            'train_period': (str(train['dispatch_date'].min()), str(train['dispatch_date'].max())) if 'dispatch_date' in train else None,
            'test_period': (str(test['dispatch_date'].min()), str(test['dispatch_date'].max())) if 'dispatch_date' in test else None}
    return Evaluation(final, holdout_model, best, comparison, metrics, tuned, random_split, leakage_check, curves,
                      importance, scores.rename('delay_risk'), X_te.index, meta)


def train_and_evaluate(df: pd.DataFrame):
    """Backward-compatible API: (model refit on all rows, holdout metrics dict)."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[TARGET].nunique() < 2:
        raise ValueError('Need both target classes')
    ev = evaluate(df)
    return ev.model, {**ev.metrics, 'model': ev.model_name, 'threshold': ev.meta['threshold']}


# ---------------------------------------------------------------- scoring
def risk_band(p) -> pd.Categorical:
    lo, hi = RISK_BANDS
    return pd.cut(p, [-0.001, lo, hi, 1.0], labels=['Low', 'Medium', 'High'])


def score_shipments(pipeline: Pipeline, df: pd.DataFrame, scores: pd.Series | None = None) -> pd.DataFrame:
    """Attach predicted delay probability and risk band, sorted by risk.

    Pass ``scores`` (from ``Evaluation.scores``) when scoring the training data itself, so no row is
    scored by a model that saw its outcome.
    """
    if not set(FEATURES).issubset(df.columns):
        raise ValueError('Required feature columns missing')
    result = df.copy()
    result['delay_risk'] = (scores.reindex(result.index).to_numpy() if scores is not None
                            else pipeline.predict_proba(result[FEATURES])[:, 1])
    result['risk_band'] = risk_band(result['delay_risk'])
    return result.sort_values('delay_risk', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- persistence
def save_model(ev: Evaluation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': ev.model, 'model_name': ev.model_name, 'features': FEATURES, 'threshold': ev.meta['threshold'],
                 'risk_bands': RISK_BANDS, 'sklearn_version': sklearn.__version__,
                 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                 'holdout_metrics': {k: v for k, v in ev.metrics.items() if k != 'report'}}, path)
    return path


def load_model(path: str | Path) -> dict:
    """Load a bundle saved by save_model. Only load files you created yourself (joblib can run code)."""
    bundle = joblib.load(path)
    if bundle.get('features') != FEATURES:
        raise ValueError('Saved model expects different input features')
    if bundle.get('sklearn_version') != sklearn.__version__:
        bundle['warning'] = f"Saved with scikit-learn {bundle.get('sklearn_version')}, running {sklearn.__version__}; retrain if predictions look wrong."
    return bundle
