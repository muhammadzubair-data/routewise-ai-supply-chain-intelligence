"""Synthetic supply-chain delay classification; no real shipment data."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['distance_km','planned_days','weight_kg','weather_severity','traffic_index','warehouse_backlog','carrier_on_time_rate','customs_hold_hours']
CATEGORICAL = ['transport_mode','region','priority']
FEATURES = NUMERIC + CATEGORICAL


def generate_data(n: int = 3500, seed: int = 42) -> pd.DataFrame:
    if n < 100: raise ValueError('n must be >= 100')
    rng = np.random.default_rng(seed)
    mode = rng.choice(['Road','Rail','Air','Sea'], n, p=[.58,.17,.13,.12])
    region = rng.choice(['North','South','East','West','Central'], n)
    priority = rng.choice(['Standard','Express','Critical'], n, p=[.68,.25,.07])
    distance = np.round(rng.uniform(20, 3000, n), 1)
    planned = np.maximum(1, np.ceil(distance / np.where(mode=='Air', 1250, np.where(mode=='Rail',550,np.where(mode=='Sea',350,400))))).astype(int)
    weather = rng.integers(0,6,n)
    traffic = rng.integers(10,101,n)
    backlog = rng.integers(0,101,n)
    ontime = np.round(rng.uniform(.65,.99,n),3)
    customs = rng.choice([0,0,0,0,4,8,16,24,36],n)
    weight = np.round(rng.lognormal(5.0, 1.0, n),1)
    logit = (-3.0 + .44*weather + .023*traffic + .018*backlog - 4.0*(ontime-.75) + .035*customs + .00022*distance + np.where(mode=='Sea',.35,0) - np.where(priority=='Critical',.5,0))
    p = 1/(1+np.exp(-logit))
    delayed = rng.binomial(1,p)
    return pd.DataFrame({'shipment_id':[f'RW-{i:06d}' for i in range(1,n+1)], 'distance_km':distance,'planned_days':planned,'weight_kg':weight,'weather_severity':weather,'traffic_index':traffic,'warehouse_backlog':backlog,'carrier_on_time_rate':ontime,'customs_hold_hours':customs,'transport_mode':mode,'region':region,'priority':priority,'delayed':delayed})


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median'))]),NUMERIC),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)
    ])
    return Pipeline([('preprocess',pre),('model',RandomForestClassifier(n_estimators=160,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1))])


def train_and_evaluate(df: pd.DataFrame):
    missing = set(FEATURES + ['delayed']) - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['delayed'].nunique() < 2: raise ValueError('Need both target classes')
    x_train,x_test,y_train,y_test = train_test_split(df[FEATURES],df['delayed'],test_size=.25,stratify=df['delayed'],random_state=42)
    pipeline=build_pipeline().fit(x_train,y_train)
    prob=pipeline.predict_proba(x_test)[:,1]
    pred=(prob>=.5).astype(int)
    metrics={'roc_auc':float(roc_auc_score(y_test,prob)),'average_precision':float(average_precision_score(y_test,prob)),'report':classification_report(y_test,pred,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'test_rows':len(y_test),'train_rows':len(x_train),'positive_rate':float(y_test.mean())}
    return pipeline,metrics


def score_shipments(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    if not set(FEATURES).issubset(df.columns): raise ValueError('Required feature columns missing')
    result=df.copy()
    result['delay_risk']=pipeline.predict_proba(df[FEATURES])[:,1]
    result['risk_band']=pd.cut(result['delay_risk'],bins=[-0.001,.3,.6,1],labels=['Low','Medium','High'])
    return result.sort_values('delay_risk',ascending=False).reset_index(drop=True)
