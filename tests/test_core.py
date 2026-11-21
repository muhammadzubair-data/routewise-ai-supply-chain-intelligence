import numpy as np
import pytest
from routewise.core import generate_data,train_and_evaluate,score_shipments,FEATURES

def test_reproducible():
    assert generate_data(120,7).equals(generate_data(120,7))
def test_schema():
    d=generate_data(120); assert set(FEATURES+['delayed']).issubset(d.columns)
def test_binary():
    assert set(generate_data(150).delayed.unique()).issubset({0,1})
def test_minimum_rows():
    with pytest.raises(ValueError): generate_data(99)
def test_training_and_scoring():
    d=generate_data(300); model,m=train_and_evaluate(d)
    scored=score_shipments(model,d)
    assert len(scored)==300 and scored.delay_risk.between(0,1).all()
    assert 0<=m['roc_auc']<=1 and m['test_rows']==75
    assert np.all(np.diff(scored.delay_risk.to_numpy())<=0)
def test_missing_features():
    d=generate_data(120).drop(columns=['distance_km'])
    with pytest.raises(ValueError): train_and_evaluate(d)
