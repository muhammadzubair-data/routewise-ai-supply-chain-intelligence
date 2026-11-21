"""Generator realism, leakage controls, temporal evaluation, validation, persistence and app start-up."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from routewise.core import (
    FEATURES,
    ID_COLUMNS,
    OUTCOMES,
    RISK_BANDS,
    evaluate,
    generate_data,
    load_model,
    risk_band,
    save_model,
    score_shipments,
    temporal_split,
    validate_input,
    validate_shipments,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def data():
    return generate_data(3000, 11)


@pytest.fixture(scope='module')
def ev(data):
    return evaluate(data)


# ------------------------------------------------------------ synthetic data
def test_generator_realism(data):
    assert 0.12 < data.delayed.mean() < 0.30
    dates = pd.to_datetime(data.dispatch_date)
    assert dates.is_monotonic_increasing and dates.dt.year.eq(2025).all()
    assert (data.actual_transit_days == data.planned_days + data.delay_days).all()
    assert ((data.delay_days > 0) == (data.delayed == 1)).all()
    assert (data.loc[data.customs_clearance_required == 0, 'customs_hold_hours'] == 0).all()
    assert data[['traffic_index', 'carrier_on_time_rate']].isna().sum().sum() > 0   # realistic gaps


def test_sample_csv_matches_generator():
    sample = pd.read_csv(ROOT / 'sample_data' / 'demo_shipments.csv')
    expected = generate_data(150, 42, missing_rate=0)
    expected['dispatch_date'] = expected['dispatch_date'].astype(str)
    pd.testing.assert_frame_equal(sample, expected, check_dtype=False)


# ------------------------------------------------------------ leakage & split
def test_no_outcome_or_id_in_features():
    assert not set(FEATURES) & set(OUTCOMES + ID_COLUMNS)


def test_model_inputs_are_dispatch_time_only(ev):
    pre = ev.model.named_steps['preprocess']
    used = [c for _, _, cols in pre.transformers_ if isinstance(cols, list) for c in cols]
    assert set(used) == set(FEATURES)


def test_temporal_split_is_forward_in_time(data):
    train, test = temporal_split(data)
    assert len(test) == 750
    assert pd.to_datetime(train.dispatch_date).max() <= pd.to_datetime(test.dispatch_date).min()


# ------------------------------------------------------------ evaluation protocol
def test_evaluation_outputs(ev, data):
    assert set(ev.comparison.Model) == {'Random Forest (balanced)', 'Random Forest (unweighted)', 'Logistic Regression'}
    assert ev.model_name == ev.comparison.iloc[0].Model                      # selected by CV, not holdout
    m = ev.metrics
    assert m['test_rows'] == 750 and m['train_rows'] == 2250
    assert m['average_precision'] > m['positive_rate'] * 1.5                 # clearly better than no skill
    assert 0.6 < m['roc_auc'] <= 1 and 0 < ev.meta['threshold'] < 1
    assert {'roc_auc', 'average_precision'} <= set(ev.random_split)
    assert ev.leakage_check['leaky_average_precision'] > 0
    assert set(ev.importance.Feature) == set(FEATURES)


def test_scores_never_in_sample(ev, data):
    clean, _ = validate_shipments(data)
    assert ev.scores.index.equals(clean.index) and ev.scores.notna().all()
    holdout_scores = ev.holdout_model.predict_proba(clean.loc[ev.holdout_index, FEATURES])[:, 1]
    assert np.allclose(ev.scores.loc[ev.holdout_index], holdout_scores)
    s = score_shipments(ev.model, clean, ev.scores)
    assert s.delay_risk.is_monotonic_decreasing and s.delay_risk.between(0, 1).all()


def test_reproducible():
    a, b = evaluate(generate_data(1200, 3)), evaluate(generate_data(1200, 3))
    assert a.metrics['roc_auc'] == b.metrics['roc_auc'] and a.model_name == b.model_name


def test_risk_bands():
    lo, hi = RISK_BANDS
    assert list(risk_band(pd.Series([lo - .01, (lo + hi) / 2, hi + .01])).astype(str)) == ['Low', 'Medium', 'High']


# ------------------------------------------------------------ validation
def test_validation_drops_invalid_and_keeps_missing(data):
    bad = data.head(50).copy()
    bad.loc[bad.index[0], 'transport_mode'] = 'Teleport'
    bad.loc[bad.index[1], 'weather_severity'] = 9
    bad.loc[bad.index[2], 'traffic_index'] = np.nan
    bad.loc[bad.index[3], 'region'] = ' north '                       # normalised, kept
    clean, report = validate_shipments(bad)
    assert len(clean) == 48 and report.dropped_rows == 2
    assert clean['traffic_index'].isna().sum() >= 1 and 'North' in set(clean.region)


def test_validation_missing_column(data):
    with pytest.raises(ValueError, match='Missing columns'):
        validate_shipments(data.drop(columns=['priority']))


def _good():
    return {'distance_km': 800, 'planned_days': 2, 'weight_kg': 120, 'weather_severity': 2, 'traffic_index': 50,
            'warehouse_backlog': 40, 'carrier_on_time_rate': .9, 'customs_clearance_required': 0,
            'transport_mode': 'Road', 'region': 'North', 'priority': 'Standard'}


@pytest.mark.parametrize('change,msg', [({'weather_severity': 8}, 'weather_severity'), ({'transport_mode': 'Drone'}, 'transport_mode'),
                                        ({'carrier_on_time_rate': 1.4}, 'carrier_on_time_rate'), ({'distance_km': None}, 'distance_km is required')])
def test_validate_input_errors(change, msg):
    with pytest.raises(ValueError, match=msg):
        validate_input({**_good(), **change})


def test_prediction_responds_to_risk_factors(ev):
    calm = ev.model.predict_proba(validate_input(_good()))[0, 1]
    storm = ev.model.predict_proba(validate_input({**_good(), 'weather_severity': 5, 'traffic_index': 95, 'warehouse_backlog': 95}))[0, 1]
    assert storm > calm


# ------------------------------------------------------------ persistence & CLI
def test_save_and_load_roundtrip(ev, tmp_path):
    path = save_model(ev, tmp_path / 'm.joblib')
    bundle = load_model(path)
    x = validate_input(_good())
    assert bundle['features'] == FEATURES and 0 < bundle['threshold'] < 1
    assert bundle['model'].predict_proba(x)[0, 1] == pytest.approx(ev.model.predict_proba(x)[0, 1])


def test_load_rejects_mismatched_features(ev, tmp_path):
    import joblib
    path = save_model(ev, tmp_path / 'm.joblib')
    b = joblib.load(path)
    b['features'] = ['other']
    joblib.dump(b, path)
    with pytest.raises(ValueError, match='different input features'):
        load_model(path)


def test_train_script(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from train import main
    summary = main(['--n', '1500', '--out', str(tmp_path)])
    saved = json.loads((tmp_path / 'metrics.json').read_text())
    assert saved['model'] == summary['model'] and (tmp_path / 'routewise_model.joblib').exists()


# ------------------------------------------------------------ app
def test_app_starts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('Synthetic data' in w.value for w in at.warning)
