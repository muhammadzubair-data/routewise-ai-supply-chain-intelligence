"""RouteWise AI | synthetic demonstration only."""
import streamlit as st
import pandas as pd
import plotly.express as px
from routewise.core import generate_data,train_and_evaluate,score_shipments,FEATURES

st.set_page_config(page_title='RouteWise AI | ZubairLabs',page_icon='🚚',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1350px}div[data-testid="stMetric"]{background:#111d32;border:1px solid #263b58;padding:14px;border-radius:14px}div[data-testid="stMetric"] *{color:#e8f4ff!important}.hero{background:linear-gradient(115deg,#0b1930,#103c57 65%,#19565e);border:1px solid #1c6b86;border-radius:20px;padding:27px;margin-bottom:18px;color:white}.hero h1{color:white;margin:0}.hero p{color:#b9e8f1}</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚚 RouteWise AI</h1><p>Supply Chain Delay & Delivery Risk Intelligence · ZubairLabs</p><b>DEMONSTRATION · SYNTHETIC DATA · NOT LIVE LOGISTICS TELEMETRY</b></div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo(n,seed): return generate_data(n,seed)
@st.cache_resource(show_spinner='Training demonstration model...')
def fit_demo(n,seed):
    df=generate_data(n,seed)
    return train_and_evaluate(df)

with st.sidebar:
    st.header('⚙️ Demo settings')
    n=st.slider('Synthetic shipments',500,10000,3500,500)
    seed=st.number_input('Reproducibility seed',min_value=0,max_value=999999,value=42)
    st.caption('Model trained on a stratified 75/25 split. All records are generated locally.')
    st.divider()
    st.markdown('**ZubairLabs** · [GitHub](https://github.com/muhammadzubair-data/routewise-ai-supply-chain-intelligence)')

df=load_demo(n,seed)
model,metrics=fit_demo(n,seed)
scored=score_shipments(model,df)
tabs=st.tabs(['📊 Command Center','🚨 Risk Explorer','🔮 What-if Simulator','🧠 Model Evaluation','📁 Data & Exports','ℹ️ Methodology'])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric('Synthetic shipments',f'{len(df):,}')
    b.metric('Observed synthetic delay rate',f'{df.delayed.mean():.1%}')
    c.metric('High-risk shipments',f'{(scored.risk_band=="High").sum():,}')
    d.metric('Held-out ROC-AUC',f'{metrics["roc_auc"]:.3f}')
    l,r=st.columns(2)
    with l: st.plotly_chart(px.histogram(scored,x='risk_band',color='risk_band',title='Risk distribution',category_orders={'risk_band':['Low','Medium','High']}),use_container_width=True)
    with r: st.plotly_chart(px.bar(df.groupby('transport_mode',as_index=False)['delayed'].mean(),x='transport_mode',y='delayed',title='Synthetic delay rate by transport mode',labels={'delayed':'Delay rate'}),use_container_width=True)
    st.plotly_chart(px.scatter(scored.sample(min(700,len(scored)),random_state=42),x='distance_km',y='delay_risk',color='transport_mode',hover_data=['shipment_id','priority'],title='Distance vs predicted delay risk'),use_container_width=True)
with tabs[1]:
    regions=st.multiselect('Regions',sorted(df.region.unique()),default=sorted(df.region.unique()))
    bands=st.multiselect('Risk bands',['Low','Medium','High'],default=['High','Medium'])
    filtered=scored[scored.region.isin(regions)&scored.risk_band.astype(str).isin(bands)]
    st.dataframe(filtered[['shipment_id','region','transport_mode','priority','distance_km','weather_severity','traffic_index','warehouse_backlog','delay_risk','risk_band']],use_container_width=True,hide_index=True)
    st.download_button('Download filtered risk report (CSV)',filtered.to_csv(index=False).encode(),file_name='routewise_risk_report.csv',mime='text/csv')
with tabs[2]:
    st.info('Change pre-dispatch operating conditions to explore the trained demonstration model. This is not a validated production forecast.')
    c1,c2,c3=st.columns(3)
    with c1:
        distance=st.slider('Distance (km)',20,3000,700)
        planned=st.slider('Planned transit days',1,15,3)
        weight=st.number_input('Weight (kg)',min_value=1.0,value=150.0)
        weather=st.slider('Weather severity (0–5)',0,5,2)
    with c2:
        traffic=st.slider('Traffic index',10,100,50)
        backlog=st.slider('Warehouse backlog index',0,100,40)
        ontime=st.slider('Carrier historical on-time rate',0.65,0.99,0.88,0.01)
        customs=st.slider('Known customs hold (hours)',0,36,0)
    with c3:
        mode=st.selectbox('Transport mode',['Road','Rail','Air','Sea'])
        region=st.selectbox('Region',['North','South','East','West','Central'])
        priority=st.selectbox('Priority',['Standard','Express','Critical'])
    sample=pd.DataFrame([dict(distance_km=distance,planned_days=planned,weight_kg=weight,weather_severity=weather,traffic_index=traffic,warehouse_backlog=backlog,carrier_on_time_rate=ontime,customs_hold_hours=customs,transport_mode=mode,region=region,priority=priority)])
    risk=float(model.predict_proba(sample[FEATURES])[:,1][0])
    st.metric('Estimated synthetic-model delay probability',f'{risk:.1%}')
    st.progress(risk)
with tabs[3]:
    st.subheader('Held-out evaluation · synthetic observations')
    a,b,c=st.columns(3)
    a.metric('ROC-AUC',f'{metrics["roc_auc"]:.3f}')
    b.metric('Average precision',f'{metrics["average_precision"]:.3f}')
    c.metric('Test observations',f'{metrics["test_rows"]:,}')
    cm=pd.DataFrame(metrics['confusion_matrix'],index=['Actual on time','Actual delayed'],columns=['Predicted on time','Predicted delayed'])
    st.dataframe(cm,use_container_width=True)
    st.dataframe(pd.DataFrame(metrics['report']).transpose(),use_container_width=True)
    st.caption('The training data is generated using a known simulated relationship. Metrics do not generalize to real-world logistics without external validation.')
with tabs[4]:
    st.dataframe(df.head(500),use_container_width=True,hide_index=True)
    st.download_button('Download synthetic dataset',df.to_csv(index=False).encode(),file_name='routewise_synthetic_shipments.csv',mime='text/csv')
    st.download_button('Download all scored shipments',scored.to_csv(index=False).encode(),file_name='routewise_scored_shipments.csv',mime='text/csv')
with tabs[5]:
    st.markdown('''### Responsible demonstration
- **Source:** deterministic synthetic generator (`routewise/core.py`). No third-party operational data or API keys.
- **Label:** a Bernoulli draw from a documented simulated logistic relationship; not real delivery outcomes.
- **Training:** Random Forest, one-hot encoding, stratified 75/25 holdout, fixed seed.
- **Leakage control:** model excludes `shipment_id` and the `delayed` label. Input variables represent assumed pre-dispatch or known operational conditions.
- **Caution:** probability scores are not calibrated or externally validated. This application does not provide live tracking, route optimization, or guaranteed delivery estimates.
- **Next:** validate with consented operational data, perform temporal splits, calibrate probabilities, assess fairness across routes/carriers, and monitor drift.''')
