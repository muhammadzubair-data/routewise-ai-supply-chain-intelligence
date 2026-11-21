"""RouteWise AI — Supply Chain Delay & Delivery Risk Intelligence (Streamlit app).

Synthetic demonstration data only. Run locally: streamlit run app.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from routewise.core import (
    CATEGORY_VALUES,
    FEATURE_INFO,
    FEATURES,
    ID_COLUMNS,
    NUMERIC,
    OUTCOMES,
    RANGES,
    RISK_BANDS,
    TARGET,
    Evaluation,
    evaluate,
    generate_data,
    risk_band,
    score_shipments,
    validate_input,
    validate_shipments,
)

WEBSITE = 'https://zubairlabs.com'
GITHUB = 'https://github.com/muhammadzubair-data/routewise-ai-supply-chain-intelligence'
ASSETS = Path(__file__).parent / 'assets'
NAVY, CYAN, TEAL, BLUE, AMBER, RED, GREEN = '#0A1A33', '#22D3EE', '#14B8A6', '#2563EB', '#F59E0B', '#EF4444', '#10B981'
BAND_COLORS = {'Low': TEAL, 'Medium': AMBER, 'High': RED}
OBS_COLORS = {'On time (observed)': 'rgba(37,99,235,0.35)', 'Delayed (observed)': RED}

st.set_page_config(page_title='RouteWise AI | ZubairLabs', page_icon='🚚', layout='wide',
                   menu_items={'About': f'RouteWise AI · ZubairLabs portfolio project · synthetic data · {WEBSITE}'})
st.markdown(f"""<style>
.block-container {{max-width: 1440px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.rw-hero {{padding: 24px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: radial-gradient(900px 260px at 0% 0%, rgba(34,211,238,.28), transparent 60%),
              radial-gradient(700px 260px at 100% 100%, rgba(20,184,166,.30), transparent 60%),
              linear-gradient(120deg, #06132A 0%, #0A1A33 55%, #0B2E4A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.rw-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: {CYAN}; text-transform: uppercase;}}
.rw-hero h1 {{font-size: 32px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.rw-hero p {{margin: 0; color: #CBD5E1;}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.5); color: #FDE68A;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.28);
  background: linear-gradient(160deg, rgba(34,211,238,.10), rgba(20,184,166,.06));}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12px; opacity: .65; margin-top: 4px;}}
.sec-title {{font-size: 24px; font-weight: 800; margin: 2px 0;}}
.sec-sub {{opacity: .7; margin-bottom: 14px; font-size: 14px;}}
.note {{border-left: 4px solid {TEAL}; padding: 10px 14px; border-radius: 10px; background: rgba(20,184,166,.09); margin: 6px 0 12px 0;}}
.result {{border-radius: 18px; padding: 20px 22px; border: 1px solid rgba(34,211,238,.45);
  background: linear-gradient(135deg, rgba(34,211,238,.14), rgba(37,99,235,.12));}}
.result .big {{font-size: 42px; font-weight: 800; line-height: 1.05;}}
.footer {{opacity: .6; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def kpis(items):
    e = html.escape
    st.markdown("<div class='kpi-grid'>" + ''.join(
        f"<div class='kpi'><div class='label'>{e(a)}</div><div class='value'>{e(b)}</div><div class='sub'>{e(c)}</div></div>"
        for a, b, c in items) + '</div>', unsafe_allow_html=True)


def section(title, sub):
    st.markdown(f"<div class='sec-title'>{html.escape(title)}</div><div class='sec-sub'>{html.escape(sub)}</div>",
                unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text='', title_font=dict(size=15))
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False})


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v)
    return out.to_csv(index=False).encode()


def pct(x):
    return f'{x:.1%}'


# ------------------------------------------------------------------ data + model (cached)
@st.cache_data(show_spinner=False, max_entries=6)
def load(n: int, seed: int):
    raw = generate_data(n, seed)
    clean, report = validate_shipments(raw)
    return raw, clean, report


@st.cache_resource(show_spinner=False, max_entries=4)
def fit(n: int, seed: int) -> Evaluation:
    return evaluate(load(n, seed)[1])


st.logo(str(ASSETS / 'logo.svg'), icon_image=str(ASSETS / 'icon.svg'), link=WEBSITE, size='large')
with st.sidebar:
    st.markdown('**Synthetic dataset**')
    n = st.select_slider('Shipments', options=[2000, 4000, 6000, 8000, 10000, 15000, 20000], value=6000)
    seed = int(st.number_input('Seed', 0, 999_999, 42, help='Same seed → identical data and results'))

raw, data, quality = load(n, seed)
with st.spinner('Training and evaluating models (cached after first run)…'):
    ev = fit(n, seed)
scored = score_shipments(ev.model, data, ev.scores)   # never-in-sample scores
scored['dispatch_month'] = pd.to_datetime(scored['dispatch_date']).dt.to_period('M').dt.to_timestamp()
scored['outcome'] = scored[TARGET].map({0: 'On time (observed)', 1: 'Delayed (observed)'})
scored['period'] = np.where(scored['shipment_id'].isin(data.loc[ev.holdout_index, 'shipment_id']), 'Holdout (future)', 'Training period')

with st.sidebar:
    st.markdown('**Filters**')
    f_mode = st.multiselect('Transport mode', CATEGORY_VALUES['transport_mode'], default=CATEGORY_VALUES['transport_mode'])
    f_region = st.multiselect('Region', CATEGORY_VALUES['region'], default=CATEGORY_VALUES['region'])
    f_priority = st.multiselect('Priority', CATEGORY_VALUES['priority'], default=CATEGORY_VALUES['priority'])
    st.caption('🌗 Theme: ⋮ menu → Settings → Theme')
    st.markdown(f'[🌐 zubairlabs.com]({WEBSITE}) · [GitHub]({GITHUB})')

view = scored[scored.transport_mode.isin(f_mode) & scored.region.isin(f_region) & scored.priority.isin(f_priority)]

st.markdown(f"""<div class='rw-hero'><div class='eyebrow'>ZubairLabs · Supply Chain Intelligence</div>
<h1>🚚 RouteWise AI</h1><p>Delivery delay prediction · shipment risk analytics · what-if simulation</p>
<span class='pill warn'>⚠ Synthetic demonstration data</span><span class='pill'>{len(view):,} of {len(scored):,} shipments in view</span>
<span class='pill'>Model: {html.escape(ev.model_name)}</span><span class='pill'>Prediction moment: dispatch</span></div>""",
            unsafe_allow_html=True)
st.warning('**Synthetic data, not live logistics telemetry.** Shipments and outcomes are simulated from a documented formula. '
           'There is no carrier API, GPS feed or real tracking. Predicted probabilities are model estimates; observed outcomes are the simulated labels.',
           icon='⚠️')


def need_rows():
    if view.empty:
        st.info('No shipments match the sidebar filters.')
        return False
    return True


# ------------------------------------------------------------------ pages
def page_command():
    section('Command Center', 'Volume, observed delays and predicted risk across the 2025 synthetic shipment history.')
    if not need_rows():
        return
    high = (view.risk_band == 'High').sum()
    kpis([
        ('Shipments', f'{len(view):,}', 'dispatched in 2025 (synthetic)'),
        ('Observed delay rate', pct(view[TARGET].mean()), 'actual simulated outcomes'),
        ('Mean predicted risk', pct(view.delay_risk.mean()), 'model estimate (never in-sample)'),
        ('High-risk shipments', f'{high:,}', f'predicted ≥ {RISK_BANDS[1]:.0%}'),
        ('Holdout PR-AUC', f"{ev.metrics['average_precision']:.3f}", f"baseline {ev.metrics['positive_rate']:.3f}"),
    ])
    monthly = view.groupby('dispatch_month').agg(shipments=(TARGET, 'size'), observed=(TARGET, 'mean'), predicted=('delay_risk', 'mean')).reset_index()
    fig = go.Figure()
    fig.add_bar(x=monthly.dispatch_month, y=monthly.shipments, name='Shipments', marker_color='rgba(37,99,235,0.35)', yaxis='y2')
    fig.add_scatter(x=monthly.dispatch_month, y=monthly.observed, name='Observed delay rate', mode='lines+markers', line=dict(color=RED, width=3))
    fig.add_scatter(x=monthly.dispatch_month, y=monthly.predicted, name='Mean predicted risk', mode='lines+markers', line=dict(color=CYAN, width=3, dash='dot'))
    fig.update_layout(title='Monthly volume, observed delays and predicted risk', yaxis=dict(tickformat='.0%', title='Rate'),
                      yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Shipments'), legend=dict(orientation='h', y=-0.15))
    if ev.meta['test_period']:
        fig.add_vrect(x0=ev.meta['test_period'][0], x1=ev.meta['test_period'][1], fillcolor='rgba(20,184,166,0.08)', line_width=0,
                      annotation_text='holdout period', annotation_position='top left')
    chart(fig, 400)
    c1, c2 = st.columns(2)
    with c1:
        bands = view.risk_band.value_counts().reindex(['Low', 'Medium', 'High']).fillna(0).reset_index()
        bands.columns = ['Risk band', 'Shipments']
        chart(px.pie(bands, names='Risk band', values='Shipments', hole=.6, color='Risk band', color_discrete_map=BAND_COLORS,
                     title='Predicted risk bands'))
    with c2:
        by_band = view.groupby('risk_band', observed=True).agg(predicted=('delay_risk', 'mean'), observed=(TARGET, 'mean')).reset_index()
        long = by_band.melt('risk_band', var_name='Measure', value_name='Rate')
        long['Measure'] = long['Measure'].map({'predicted': 'Mean predicted risk', 'observed': 'Observed delay rate'})
        fig = px.bar(long, x='risk_band', y='Rate', color='Measure', barmode='group', title='Predicted vs observed, by risk band',
                     color_discrete_map={'Mean predicted risk': CYAN, 'Observed delay rate': RED}, labels={'risk_band': ''})
        fig.update_layout(yaxis_tickformat='.0%')
        chart(fig)
    st.markdown('##### Operational drivers (observed delay rate)')
    c3, c4, c5 = st.columns(3)
    for col, (key, title) in zip([c3, c4, c5], [('transport_mode', 'By transport mode'), ('priority', 'By priority'),
                                                ('weather_severity', 'By weather forecast severity')]):
        with col:
            g = view.groupby(key, observed=True).agg(rate=(TARGET, 'mean'), n=(TARGET, 'size')).reset_index()
            fig = px.bar(g, x=key, y='rate', text=g['rate'].map(pct), color_discrete_sequence=[TEAL], title=title,
                         labels={'rate': 'Observed delay rate', key: ''}, hover_data={'n': True})
            fig.update_layout(yaxis_tickformat='.0%')
            chart(fig, 330)
    dist = view.assign(distance_band=pd.cut(view.distance_km, [0, 500, 1000, 1500, 2000, 3100],
                                            labels=['<500 km', '500–1,000', '1,000–1,500', '1,500–2,000', '2,000+']))
    g = dist.groupby(['distance_band', 'customs_clearance_required'], observed=True)[TARGET].mean().reset_index()
    g['Customs'] = g['customs_clearance_required'].map({0: 'No customs', 1: 'Customs required'})
    fig = px.bar(g, x='distance_band', y=TARGET, color='Customs', barmode='group', title='Observed delay rate by distance and customs requirement',
                 color_discrete_map={'No customs': BLUE, 'Customs required': AMBER}, labels={TARGET: 'Observed delay rate', 'distance_band': ''})
    fig.update_layout(yaxis_tickformat='.0%')
    chart(fig, 360)
    worst = view.groupby(['transport_mode', 'region'], observed=True)[TARGET].mean().sort_values(ascending=False)
    note(f"<b>In this synthetic data:</b> the highest observed delay rate is <b>{html.escape(worst.index[0][0])} → {html.escape(worst.index[0][1])}</b> "
         f"({pct(worst.iloc[0])}); weather forecast severity 4–5 shipments are delayed {pct(view[view.weather_severity >= 4][TARGET].mean())} of the time "
         f"vs {pct(view[view.weather_severity <= 1][TARGET].mean())} at severity 0–1. These patterns come from the simulation formula.")


def _form(prefix: str, defaults: dict) -> dict:
    c1, c2, c3 = st.columns(3)
    rec = {}
    rec['distance_km'] = c1.number_input('Distance (km)', 1.0, 20000.0, float(defaults['distance_km']), 10.0, key=f'{prefix}_dist')
    rec['planned_days'] = c1.number_input('Planned transit days', 1, 60, int(defaults['planned_days']), key=f'{prefix}_days')
    rec['weight_kg'] = c1.number_input('Weight (kg)', 0.1, 50000.0, float(defaults['weight_kg']), 10.0, key=f'{prefix}_wt')
    rec['transport_mode'] = c1.selectbox('Transport mode', CATEGORY_VALUES['transport_mode'],
                                         index=CATEGORY_VALUES['transport_mode'].index(defaults['transport_mode']), key=f'{prefix}_mode')
    rec['weather_severity'] = c2.slider('Weather forecast severity (0–5)', 0, 5, int(defaults['weather_severity']), key=f'{prefix}_wx')
    rec['traffic_index'] = c2.slider('Traffic forecast index', 0, 100, int(defaults['traffic_index']), key=f'{prefix}_tr')
    rec['warehouse_backlog'] = c2.slider('Warehouse backlog index', 0, 100, int(defaults['warehouse_backlog']), key=f'{prefix}_bl')
    rec['region'] = c2.selectbox('Destination region', CATEGORY_VALUES['region'], index=CATEGORY_VALUES['region'].index(defaults['region']), key=f'{prefix}_reg')
    rec['carrier_on_time_rate'] = c3.slider('Carrier historical on-time rate', 0.0, 1.0, float(defaults['carrier_on_time_rate']), 0.01, key=f'{prefix}_ot')
    rec['customs_clearance_required'] = int(c3.toggle('Customs clearance required', bool(defaults['customs_clearance_required']), key=f'{prefix}_cus'))
    rec['priority'] = c3.selectbox('Priority', CATEGORY_VALUES['priority'], index=CATEGORY_VALUES['priority'].index(defaults['priority']), key=f'{prefix}_pri')
    return rec


def _reference() -> dict:
    d = data[FEATURES]
    ref = {c: float(d[c].median()) for c in NUMERIC}
    ref.update({c: d[c].mode().iloc[0] for c in CATEGORY_VALUES})
    ref['customs_clearance_required'] = 0
    ref['planned_days'] = int(ref['planned_days'])
    return ref


def _predict(rec: dict) -> float:
    return float(ev.model.predict_proba(validate_input(rec)[FEATURES])[0, 1])


def page_predict():
    section('Delay Prediction', 'Estimate the probability that a shipment will arrive late, using only information known at dispatch.')
    ref = _reference()
    defaults = {**ref, 'distance_km': 1200.0, 'planned_days': 3, 'weather_severity': 3, 'traffic_index': 70, 'warehouse_backlog': 60,
                'carrier_on_time_rate': 0.78, 'customs_clearance_required': 1, 'transport_mode': 'Sea', 'priority': 'Standard'}
    with st.form('predict'):
        rec = _form('p', defaults)
        submitted = st.form_submit_button('Predict delay risk', type='primary', width='stretch')
    if not submitted and 'rw_pred' not in st.session_state:
        st.info('Set the shipment details and press **Predict delay risk**.', icon='🎯')
        return
    if submitted:
        st.session_state.rw_pred = rec
    rec = st.session_state.rw_pred
    try:
        p = _predict(rec)
    except ValueError as exc:
        st.error(f'Invalid input: {exc}')
        return
    band = str(risk_band(pd.Series([p])).iloc[0])
    thr = ev.meta['threshold']
    peers = data[(data.transport_mode == rec['transport_mode']) & (data.priority == rec['priority'])]
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown(f"""<div class='result'><div style='opacity:.75;font-weight:600'>Predicted delay probability</div>
<div class='big'>{p:.1%}</div><div>Risk band: <b>{band}</b> · at the tuned threshold {thr:.2f}: <b>{'flag for proactive action' if p >= thr else 'no flag'}</b></div>
<div style='opacity:.75;margin-top:8px;font-size:13px'>Observed delay rate for {html.escape(rec['transport_mode'])} / {html.escape(rec['priority'])} shipments in the data: {pct(peers[TARGET].mean()) if len(peers) else '–'} ({len(peers):,} shipments)</div></div>""",
                    unsafe_allow_html=True)
        for col in ['distance_km', 'weight_kg']:
            lo, hi = data[col].min(), data[col].max()
            if not lo <= rec[col] <= hi:
                st.warning(f'{col} = {rec[col]:g} is outside the training range {lo:g}–{hi:g}; the estimate is an extrapolation.')
    with right:
        base = _predict({**ref})
        rows = []
        for f in FEATURES:
            if rec[f] != ref[f]:
                rows.append({'Feature': FEATURE_INFO[f].split(' (')[0], 'Change vs typical shipment': _predict({**rec, f: ref[f]}) * -1 + p})
        contrib = pd.DataFrame(rows).sort_values('Change vs typical shipment') if rows else pd.DataFrame(columns=['Feature', 'Change vs typical shipment'])
        fig = px.bar(contrib, x='Change vs typical shipment', y='Feature', orientation='h',
                     color=contrib['Change vs typical shipment'] > 0, color_discrete_map={True: RED, False: TEAL},
                     title=f'What moves this estimate (vs a typical shipment at {base:.0%})')
        fig.update_layout(showlegend=False, xaxis_tickformat='+.0%')
        chart(fig, 380)
    st.caption('“What moves this estimate”: the drop in probability when each input is reset to a typical shipment’s value, one at a time. '
               'It explains the model, not real-world causes.')


def page_risk():
    section('Risk Explorer', 'Search, filter and prioritise shipments by predicted delay risk.')
    if not need_rows():
        return
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    q = c1.text_input('Search shipment ID', placeholder='e.g. RW-000123')
    bands = c2.multiselect('Risk band', ['High', 'Medium', 'Low'], default=['High', 'Medium'])
    period = c3.selectbox('Period', ['All', 'Holdout (future)', 'Training period'])
    min_risk = c4.slider('Minimum predicted risk', 0.0, 1.0, 0.0, 0.05)
    note('<b>Priority score</b> = predicted delay probability × service weight (Standard 1, Express 2, Critical 3 by default). '
         'Weights are business assumptions you can change below; they are not learned from data.')
    w1, w2, w3 = st.columns(3)
    weights = {'Standard': w1.number_input('Weight: Standard', 0.0, 10.0, 1.0, .5), 'Express': w2.number_input('Weight: Express', 0.0, 10.0, 2.0, .5),
               'Critical': w3.number_input('Weight: Critical', 0.0, 10.0, 3.0, .5)}
    v = view[view.risk_band.astype(str).isin(bands) & (view.delay_risk >= min_risk)]
    if period != 'All':
        v = v[v.period == period]
    if q.strip():
        v = v[v.shipment_id.str.contains(q.strip(), case=False, regex=False)]
    v = v.assign(priority_score=v.delay_risk * v.priority.map(weights)).sort_values('priority_score', ascending=False)
    kpis([('Shipments listed', f'{len(v):,}', 'after filters'),
          ('Mean predicted risk', pct(v.delay_risk.mean()) if len(v) else '–', 'model estimate'),
          ('Observed delay rate', pct(v[TARGET].mean()) if len(v) else '–', 'simulated outcome'),
          ('Captured delays', f"{int(v[TARGET].sum()):,}", f"of {int(view[TARGET].sum()):,} in view")])
    cols = ['shipment_id', 'dispatch_date', 'period', 'priority_score', 'delay_risk', 'risk_band', 'outcome', 'delay_days', 'transport_mode',
            'region', 'priority', 'distance_km', 'weather_severity', 'traffic_index', 'warehouse_backlog', 'carrier_on_time_rate', 'customs_clearance_required']
    st.dataframe(v[cols], hide_index=True, width='stretch', height=460, column_config={
        'delay_risk': st.column_config.ProgressColumn('Predicted risk', format='percent', min_value=0, max_value=1),
        'priority_score': st.column_config.NumberColumn('Priority score', format='%.3f'),
        'outcome': st.column_config.TextColumn('Observed outcome'),
        'delay_days': st.column_config.NumberColumn('Observed delay (days)')})
    st.download_button('⬇ Download prioritised list (CSV)', csv_bytes(v[cols]), 'routewise_risk_report.csv', 'text/csv')


def page_whatif():
    section('What-If Simulator', 'Change dispatch conditions and see how the model’s estimate responds.')
    st.info('This explores the trained model’s behaviour on synthetic data. It is not a validated forecast or an optimisation tool.', icon='🔮')
    ids = scored.sort_values('delay_risk', ascending=False).shipment_id.head(300).tolist()
    pick = st.selectbox('Start from a shipment (top 300 by predicted risk)', ['Typical shipment'] + ids)
    base = _reference() if pick == 'Typical shipment' else data[data.shipment_id == pick][FEATURES].iloc[0].to_dict()
    base = {k: (v if not (isinstance(v, float) and np.isnan(v)) else _reference()[k]) for k, v in base.items()}
    st.markdown('##### Scenario')
    scen = _form(f'w_{pick}', base)
    p0, p1 = _predict(base), _predict(scen)
    kpis([('Baseline estimate', pct(p0), pick), ('Scenario estimate', pct(p1), str(risk_band(pd.Series([p1])).iloc[0]) + ' band'),
          ('Change', f'{(p1 - p0) * 100:+.1f} pts', 'scenario − baseline')])
    c1, c2 = st.columns(2)
    with c1:
        feat = st.selectbox('Sensitivity curve for', ['weather_severity', 'traffic_index', 'warehouse_backlog', 'carrier_on_time_rate', 'distance_km'])
        lo, hi = RANGES[feat]
        lo, hi = max(lo, float(data[feat].min())), min(hi, float(data[feat].max()))
        grid = np.linspace(lo, hi, 25)
        frame = pd.DataFrame([{**scen, feat: g} for g in grid])[FEATURES]
        curve = pd.DataFrame({feat: grid, 'Predicted delay probability': ev.model.predict_proba(frame)[:, 1]})
        fig = px.line(curve, x=feat, y='Predicted delay probability', markers=True, title=f'Estimate vs {feat} (other inputs = scenario)')
        fig.add_scatter(x=[scen[feat]], y=[p1], mode='markers', marker=dict(size=14, color=RED), name='Scenario')
        fig.update_traces(line_color=CYAN, selector=dict(mode='lines+markers'))
        fig.update_layout(yaxis_tickformat='.0%')
        chart(fig, 400)
    with c2:
        rows = []
        for f in ['weather_severity', 'traffic_index', 'warehouse_backlog', 'carrier_on_time_rate', 'distance_km']:
            q10, q90 = data[f].quantile(.1), data[f].quantile(.9)
            rows.append({'Feature': f, 'Low (10th pct)': _predict({**scen, f: q10}) - p1, 'High (90th pct)': _predict({**scen, f: q90}) - p1})
        tor = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(y=tor.Feature, x=tor['Low (10th pct)'], orientation='h', name='Set to 10th percentile', marker_color=TEAL)
        fig.add_bar(y=tor.Feature, x=tor['High (90th pct)'], orientation='h', name='Set to 90th percentile', marker_color=RED)
        fig.update_layout(barmode='overlay', title='Tornado: change in estimate per feature', xaxis_tickformat='+.0%',
                          legend=dict(orientation='h', y=-0.2))
        chart(fig, 400)


def page_model():
    section('Model Performance', f"Forward-in-time holdout: shipments dispatched {ev.meta['test_period'][0]} → {ev.meta['test_period'][1]} (never used for training or tuning).")
    m, t = ev.metrics, ev.tuned
    kpis([
        ('Selected model', ev.model_name, 'best 5-fold CV PR-AUC'),
        ('ROC-AUC', f"{m['roc_auc']:.3f}", f"{m['test_rows']:,} holdout shipments"),
        ('PR-AUC', f"{m['average_precision']:.3f}", f"baseline {m['positive_rate']:.3f}"),
        (f"Recall / precision @{t['threshold']:.2f}", f"{t['recall']:.2f} / {t['precision']:.2f}", 'threshold tuned on training OOF'),
        ('Brier score', f"{m['brier']:.3f}", 'probability accuracy (lower is better)'),
    ])
    st.warning('These metrics measure how well models recover a simulated mechanism. They are not evidence of real-world delivery-delay accuracy.', icon='⚠️')
    st.markdown('##### Model comparison (5-fold CV on the training period)')
    st.dataframe(ev.comparison, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format='%.3f') for c in ev.comparison.columns if c.startswith('CV')})
    st.caption('Logistic Regression can win here because the simulated delay mechanism is itself logistic. '
               'Balanced class weights raise Random Forest recall at 0.5 but worsen probability accuracy (Brier) and PR-AUC.')
    lc = ev.leakage_check
    c0a, c0b = st.columns(2)
    with c0a:
        note(f"<b>Temporal vs random split:</b> forward-in-time ROC-AUC {m['roc_auc']:.3f} / PR-AUC {m['average_precision']:.3f}; "
             f"a random split of the same data gives {ev.random_split['roc_auc']:.3f} / {ev.random_split['average_precision']:.3f}. "
             'The temporal result is the honest one for planning future shipments.')
    with c0b:
        if lc:
            note(f"<b>Leakage check:</b> adding the post-dispatch <code>customs_hold_hours</code> would raise PR-AUC to {lc['leaky_average_precision']:.3f} "
                 f"(ROC-AUC {lc['leaky_roc_auc']:.3f}). That information does not exist at dispatch, so it is excluded.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.area(ev.curves['pr'], x='Recall', y='Precision', title='Precision–recall curve (holdout)')
        fig.add_hline(y=m['positive_rate'], line_dash='dot', annotation_text='no-skill baseline')
        fig.update_traces(line_color=CYAN, fillcolor='rgba(34,211,238,0.15)')
        chart(fig)
    with c2:
        fig = px.area(ev.curves['roc'], x='False positive rate', y='True positive rate', title='ROC curve (holdout)')
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dot', color='#94A3B8'))
        fig.update_traces(line_color=TEAL, fillcolor='rgba(20,184,166,0.15)')
        chart(fig)
    c3, c4, c5 = st.columns(3)
    mats = [(c3, [[m['tn@0.5'], m['fp@0.5']], [m['fn@0.5'], m['tp@0.5']]], 'Confusion matrix @0.50'),
            (c4, [[t['tn'], t['fp']], [t['fn'], t['tp']]], f"Confusion matrix @{t['threshold']:.2f} (tuned)")]
    for col, z, title in mats:
        with col:
            fig = px.imshow(z, text_auto=True, x=['Pred: on time', 'Pred: delayed'], y=['Obs: on time', 'Obs: delayed'],
                            color_continuous_scale=['#F8FAFC', CYAN, NAVY], title=title)
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, 360)
    with c5:
        fig = px.line(ev.curves['calibration'], x='Mean predicted probability', y='Observed delay rate', markers=True,
                      title='Calibration (holdout, 10 quantile bins)')
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dot', color='#94A3B8'))
        fig.update_traces(line_color=BLUE)
        chart(fig, 360)
    imp = ev.importance.sort_values('PR-AUC drop')
    chart(px.bar(imp, x='PR-AUC drop', y='Feature', error_x='Std', orientation='h', color_discrete_sequence=[TEAL],
                 title='Permutation importance (drop in holdout PR-AUC when shuffled)'), 420)
    with st.expander('Classification report @0.5 and methodology'):
        st.dataframe(pd.DataFrame(m['report']).T.round(3), width='stretch')
        st.markdown(f"""
- **Prediction moment:** dispatch. Inputs: {', '.join(f'`{f}`' for f in FEATURES)}.
- **Never inputs:** {', '.join(f'`{c}`' for c in OUTCOMES + ID_COLUMNS)}.
- **Split:** shipments ordered by dispatch date; latest 25% = holdout ({ev.meta['test_period'][0]} → {ev.meta['test_period'][1]}).
- **Selection:** {ev.meta['n_folds']}-fold stratified CV PR-AUC on the training period; threshold {ev.meta['threshold']:.3f} maximises F1 on out-of-fold training predictions.
- **Dashboard scores:** training-period shipments use out-of-fold predictions; holdout shipments use the training-period model.
- **Missing values:** median (numeric) / most-frequent (categorical) imputation inside the pipeline, learned from training data only.
- **Risk bands:** Low < {RISK_BANDS[0]:.0%} ≤ Medium < {RISK_BANDS[1]:.0%} ≤ High, fixed cut-offs on predicted probability chosen for readability (≈ below, around and well above the ~20% base rate).
""")


def page_data():
    section('Data Explorer', 'Inspect the synthetic shipments, data quality and exports.')
    st.markdown('##### Feature availability at prediction time')
    avail = pd.DataFrame([{'Column': c, 'Role': 'Model input (known at dispatch)', 'Meaning': FEATURE_INFO[c]} for c in FEATURES] +
                         [{'Column': c, 'Role': 'Outcome (after dispatch, never an input)', 'Meaning': d} for c, d in [
                             ('delayed', 'Arrived after planned date (0/1)'), ('delay_days', 'Days late'),
                             ('actual_transit_days', 'Planned + delay days'), ('customs_hold_hours', 'Customs hold that actually occurred')]] +
                         [{'Column': c, 'Role': 'Identifier / time key', 'Meaning': d} for c, d in [
                             ('shipment_id', 'Synthetic ID'), ('dispatch_date', 'Used only to order the temporal split')]])
    st.dataframe(avail, hide_index=True, width='stretch')
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown('##### Data quality')
        qf = quality.as_frame()
        st.dataframe(qf[qf['missing values'] > 0] if (qf['missing values'] > 0).any() else qf, hide_index=True, width='stretch',
                     column_config={'share': st.column_config.NumberColumn(format='percent')})
        st.caption(f'{quality.rows:,} rows checked · {quality.dropped_rows} dropped. Missing numeric values are imputed inside the model pipeline. '
                   + ' '.join(quality.notes))
    with c2:
        st.markdown('##### Distribution of a field')
        field_ = st.selectbox('Field', NUMERIC[:-1] + ['delay_days'])
        chart(px.histogram(view, x=field_, color='outcome', barmode='overlay', opacity=.7, nbins=40, color_discrete_map=OBS_COLORS), 320)
    st.markdown('##### Shipments')
    q = st.text_input('Filter by shipment ID or text', key='data_q')
    v = view if not q.strip() else view[view.apply(lambda r: q.strip().lower() in ' '.join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(v.drop(columns=['dispatch_month']), hide_index=True, width='stretch', height=420,
                 column_config={'delay_risk': st.column_config.ProgressColumn('Predicted risk', format='percent', min_value=0, max_value=1)})
    d1, d2, d3 = st.columns(3)
    d1.download_button('⬇ Filtered scored shipments (CSV)', csv_bytes(v.drop(columns=['dispatch_month'])), 'routewise_scored_filtered.csv', 'text/csv', width='stretch')
    d2.download_button('⬇ Full synthetic dataset (CSV)', csv_bytes(raw), 'routewise_synthetic_shipments.csv', 'text/csv', width='stretch')
    summary = {'model': ev.model_name, 'threshold': ev.meta['threshold'], 'holdout': {k: v_ for k, v_ in ev.metrics.items() if k != 'report'},
               'tuned_threshold_metrics': ev.tuned, 'random_split': ev.random_split, 'leakage_check': ev.leakage_check,
               'data': {'n': n, 'seed': seed}, 'note': 'Synthetic data; metrics do not reflect real-world accuracy.'}
    d3.download_button('⬇ Model evaluation (JSON)', json.dumps(summary, indent=2, default=str).encode(), 'routewise_evaluation.json',
                       'application/json', width='stretch')


pages = [
    st.Page(page_command, title='Command Center', icon=':material/dashboard:', url_path='command', default=True),
    st.Page(page_predict, title='Delay Prediction', icon=':material/local_shipping:', url_path='predict'),
    st.Page(page_risk, title='Risk Explorer', icon=':material/crisis_alert:', url_path='risk'),
    st.Page(page_whatif, title='What-If Simulator', icon=':material/tune:', url_path='whatif'),
    st.Page(page_model, title='Model Performance', icon=':material/insights:', url_path='model'),
    st.Page(page_data, title='Data Explorer', icon=':material/table_view:', url_path='data'),
]
st.navigation(pages, position='sidebar').run()
st.markdown(f"<div class='footer'>© 2026 ZubairLabs · RouteWise AI · synthetic demonstration data · no paid APIs · "
            f"<a href='{WEBSITE}' target='_blank'>zubairlabs.com</a></div>", unsafe_allow_html=True)
