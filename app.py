"""
Hotel Revenue Analytics - Business Intelligence Platform (Part E)
Individual BI Workshop | Streamlit + scikit-learn (NO train_test_split)

Dataset: hotel_revenue_bi.csv (432 monthly operating combinations).
Target: monthly_revenue_usd, fit on the COMPLETE dataset (in-sample metrics only).
Reproduces the workshop key: avg revenue $619,122 | R2 0.9167 | MAE 31,970 | RMSE 41,315 | scenario forecast $707,029.
"""

import datetime as _dt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Hotel Revenue Analytics", page_icon="🏨", layout="wide", initial_sidebar_state="expanded")

NUM = ['available_rooms','avg_daily_rate_usd','occupancy_rate_pct','marketing_spend_usd',
       'online_rating','competitor_price_index','booking_window_days','loyalty_members']
NUM_LABEL = {'available_rooms':'Available rooms','avg_daily_rate_usd':'Avg daily rate','occupancy_rate_pct':'Occupancy %',
             'marketing_spend_usd':'Marketing spend','online_rating':'Online rating','competitor_price_index':'Competitor index',
             'booking_window_days':'Booking window','loyalty_members':'Loyalty members'}
TARGET = 'monthly_revenue_usd'

# ---- design tokens (navy/indigo executive theme) ----
INK="#0B1220"; SLATE="#475569"; MUTED="#94A3B8"; LINE="#E6EAF0"
SURFACE="#FFFFFF"; CANVAS="#F6F8FB"; NAVY="#1E3A8A"; ACCENT="#2563EB"; TEAL="#0E7C86"; AMBER="#B45309"; GRID="#EEF2F7"

mpl.rcParams.update({
    "figure.facecolor":"none","axes.facecolor":"none","savefig.facecolor":"none",
    "axes.edgecolor":LINE,"axes.linewidth":1.0,"axes.grid":True,"grid.color":GRID,"grid.linewidth":1.0,
    "axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.titleweight":"600","axes.titlecolor":INK,
    "axes.labelsize":10,"axes.labelcolor":SLATE,"xtick.color":SLATE,"ytick.color":SLATE,"xtick.labelsize":9,"ytick.labelsize":9,
    "font.family":"sans-serif","font.sans-serif":["Inter","Segoe UI","Helvetica Neue","Arial","DejaVu Sans"]})
def style_ax(ax): ax.tick_params(length=0); ax.grid(axis="x", visible=False); return ax
def fig_close(fig): st.pyplot(fig, use_container_width=True); plt.close(fig)
def money(x): return f"${x:,.0f}"

@st.cache_data(show_spinner=False)
def load_local(path='hotel_revenue_bi.csv'): return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def train_model(df):
    X, y = df[NUM], df[TARGET]
    model = LinearRegression().fit(X, y)   # complete dataset, no train_test_split
    yp = model.predict(X)
    return model, {'r2': r2_score(y, yp), 'mae': mean_absolute_error(y, yp), 'rmse': mean_squared_error(y, yp)**0.5}, yp

@st.cache_resource(show_spinner=False)
def driver_importance(_model, df):
    imp = permutation_importance(_model, df[NUM], df[TARGET], n_repeats=10, random_state=42, scoring='r2')
    return (pd.DataFrame({'Feature': NUM, 'imp': imp.importances_mean}).sort_values('imp', ascending=False).reset_index(drop=True))

# ---- CSS ----
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --muted:{MUTED}; --line:{LINE}; --surface:{SURFACE}; --canvas:{CANVAS};
  --navy:{NAVY}; --accent:{ACCENT}; --radius:18px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
  --shadow-lg:0 2px 6px rgba(16,24,40,.06), 0 18px 48px rgba(16,24,40,.10); }}
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #EEF3FF 0%, rgba(238,243,255,0) 55%), var(--canvas); }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1340px; }}
html, body, [class*="css"] {{ font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); -webkit-font-smoothing:antialiased; }}
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden; height:0; }}
[data-testid="stToolbar"] {{ display:none; }}
::selection {{ background:{ACCENT}; color:#fff; }} ::-moz-selection {{ background:{ACCENT}; color:#fff; }}
.hero {{ position:relative; overflow:hidden; border-radius:24px; margin-bottom:22px; padding:30px 34px; color:#EAF0FF;
  background: radial-gradient(900px 300px at 90% -40%, rgba(96,140,255,.45), rgba(96,140,255,0) 60%),
    linear-gradient(135deg,#0B1220 0%,#15264F 55%,#1E3A8A 120%); box-shadow:var(--shadow-lg); }}
.hero .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; color:#9DB4FF; font-weight:700; margin-bottom:8px; }}
.hero h1 {{ font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; line-height:1.1; margin:0 0 8px; color:#fff; letter-spacing:-.01em; }}
.hero p {{ font-size:14.5px; color:#C8D4F3; margin:0; max-width:740px; line-height:1.5; }}
.hero .brand {{ position:absolute; top:22px; right:26px; display:flex; align-items:center; gap:10px; font-weight:700; color:#DBE4FF; font-size:13px; }}
.hero .brand .dot {{ width:30px;height:30px;border-radius:9px; background:linear-gradient(135deg,#3B82F6,#22D3EE); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; }}
.hero .meta {{ position:absolute; bottom:22px; right:26px; font-size:12px; color:#9DB4FF; display:flex; align-items:center; gap:8px; }}
.hero .live {{ width:8px; height:8px; border-radius:50%; background:#34D399; box-shadow:0 0 0 4px rgba(52,211,153,.18); }}
.sec {{ display:flex; align-items:center; gap:12px; margin:22px 2px 12px; }}
.sec .bar {{ width:4px; height:18px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); }}
.sec h2 {{ font-family:'Manrope',sans-serif; font-size:15px; font-weight:800; letter-spacing:.02em; text-transform:uppercase; color:var(--ink); margin:0; }}
.sec .hint {{ font-size:12.5px; color:var(--muted); margin-left:auto; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }}
@media (max-width:1100px) {{ .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{ position:relative; background:rgba(255,255,255,.80); backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s, border-color .18s; }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:var(--shadow-lg); border-color:#D5DEEC; }}
.kpi .top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi .label {{ font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--slate); }}
.kpi .ico {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:16px; background:#EEF3FF; color:var(--navy); }}
.kpi .val {{ font-family:'Manrope',sans-serif; font-size:23px; font-weight:800; color:var(--ink); margin:12px 0 4px; }}
.kpi .sub {{ font-size:12px; color:var(--muted); display:flex; gap:6px; align-items:center; }}
.kpi .chip {{ font-weight:700; padding:1px 7px; border-radius:999px; font-size:11px; }}
.chip-pos {{ color:#0F766E; background:#D7F2EE; }} .chip-warn {{ color:#92400E; background:#FBECCB; }} .chip-neu {{ color:#334155; background:#E9EEF6; }}
.kpi .accent {{ position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; background:linear-gradient(180deg,var(--accent),var(--navy)); opacity:0; transition:opacity .2s; }}
.kpi:hover .accent {{ opacity:1; }}
.ins-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
@media (max-width:1100px) {{ .ins-grid {{ grid-template-columns:1fr; }} }}
.ins {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }}
.ins:hover {{ transform:translateY(-3px); box-shadow:var(--shadow-lg); }}
.ins .k {{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }}
.ins .t {{ font-size:15px; font-weight:700; color:var(--ink); margin:0 0 6px; }}
.ins .d {{ font-size:13px; color:var(--slate); line-height:1.5; margin:0; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); padding:16px 18px 8px; height:100%; }}
.ctitle {{ font-size:14px; font-weight:700; color:var(--ink); margin-bottom:6px; }}
.note {{ font-size:12.5px; color:var(--slate); line-height:1.5; border-top:1px dashed var(--line); margin-top:2px; padding:10px 2px 2px; }}
.note b {{ color:var(--ink); }}
.fc-wrap {{ background:linear-gradient(135deg,#0B1220 0%, #15264F 100%); border-radius:22px; padding:6px; box-shadow:var(--shadow-lg); }}
.fc-value {{ background:linear-gradient(180deg,#0E1A38,#0B1220); border-radius:18px; padding:26px 24px; color:#fff; height:100%; display:flex; flex-direction:column; justify-content:center; }}
.fc-value .lab {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase; color:#9DB4FF; font-weight:700; }}
.fc-value .big {{ font-family:'Manrope',sans-serif; font-size:40px; font-weight:800; line-height:1; margin:10px 0 12px; color:#fff; }}
.fc-value .pill {{ display:inline-flex; gap:8px; align-items:center; font-size:12.5px; font-weight:600; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#DBE4FF; width:fit-content; }}
.limit {{ display:flex; gap:12px; align-items:flex-start; background:#FFF8EC; border:1px solid #F3E2BD; border-left:4px solid var(--amber); border-radius:14px; padding:14px 16px; color:#7C5310; font-size:13px; line-height:1.5; }}
section[data-testid="stSidebar"] {{ background:#0B1220; border-right:1px solid #1B2942; }}
section[data-testid="stSidebar"] * {{ color:#C8D4F3; }}
section[data-testid="stSidebar"] .side-brand {{ display:flex; gap:10px; align-items:center; padding:6px 4px 14px; border-bottom:1px solid #1B2942; margin-bottom:12px; }}
section[data-testid="stSidebar"] .side-brand .dot {{ width:30px;height:30px;border-radius:9px; background:linear-gradient(135deg,#3B82F6,#22D3EE); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; }}
section[data-testid="stSidebar"] .side-brand .n {{ font-weight:800; color:#fff; font-size:14px; }}
section[data-testid="stSidebar"] .mini {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#7E92BC; font-weight:700; margin:14px 2px 6px; }}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{ background:#101A33; border:1px solid #1B2942; border-radius:12px; padding:10px 12px; }}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color:#fff; font-size:17px; }}
.stMultiSelect [data-baseweb="tag"] {{ background:#1E3A8A !important; }}
.stButton>button {{ background:linear-gradient(135deg,var(--accent),var(--navy)); color:#fff; border:0; border-radius:12px; padding:.6rem 1.1rem; font-weight:700; box-shadow:0 8px 20px rgba(37,99,235,.28); transition:transform .15s, box-shadow .15s; }}
.stButton>button:hover {{ transform:translateY(-2px); box-shadow:0 12px 26px rgba(37,99,235,.36); }}
[data-testid="stExpander"] {{ border:1px solid var(--line) !important; border-radius:16px !important; box-shadow:var(--shadow); background:var(--surface); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color:var(--ink) !important; font-weight:700 !important; }}
.stNumberInput input {{ color:#fff !important; background:#0E1A38 !important; border-radius:10px !important; }}
.stSelectbox div[data-baseweb="select"] *, .stSelectbox div[data-baseweb="select"] {{ color:#fff !important; }}
.stNumberInput label, .stSelectbox label, .stMultiSelect label, .stSlider label {{ color:var(--ink) !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--line); flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:var(--slate); }}
.stTabs [aria-selected="true"] {{ color:var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent); }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }}
</style>
""", unsafe_allow_html=True)

# ---- sidebar ----
st.sidebar.markdown("<div class='side-brand'><div class='dot'>🏨</div>"
    "<div><div class='n'>Hotel Revenue</div><div style='font-size:11px;color:#7E92BC'>BI Platform</div></div></div>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Data source · hotel_revenue_bi.csv", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = load_local()
    except FileNotFoundError:
        st.markdown("<div class='hero'><div class='eyebrow'>Hotel Revenue</div>"
            "<h1>Revenue Analytics Platform</h1><p>Upload <b>hotel_revenue_bi.csv</b> in the sidebar to begin, "
            "or place it next to app.py.</p></div>", unsafe_allow_html=True)
        st.stop()

model, metrics, y_pred_full = train_model(df)

st.sidebar.markdown("<div class='mini'>Filters</div>", unsafe_allow_html=True)
cities = st.sidebar.multiselect("City", sorted(df.city.unique()), sorted(df.city.unique()))
segs   = st.sidebar.multiselect("Hotel segment", sorted(df.hotel_segment.unique()), sorted(df.hotel_segment.unique()))
chans  = st.sidebar.multiselect("Sales channel", sorted(df.sales_channel.unique()), sorted(df.sales_channel.unique()))
f = df[df.city.isin(cities) & df.hotel_segment.isin(segs) & df.sales_channel.isin(chans)]

st.sidebar.markdown("<div class='mini'>Model · in-sample (no split)</div>", unsafe_allow_html=True)
sm1, sm2, sm3 = st.sidebar.columns(3)
sm1.metric("R²", f"{metrics['r2']:.3f}"); sm2.metric("MAE", f"${metrics['mae']/1000:.0f}k"); sm3.metric("RMSE", f"${metrics['rmse']/1000:.0f}k")

# ---- hero ----
_now = _dt.datetime.now().strftime("%b %d, %Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="brand"><span class="dot">🏨</span> Hotel Revenue Analytics</div>
  <div class="eyebrow">Business Intelligence · Executive Report</div>
  <h1>Hotel Group — Monthly Revenue Intelligence</h1>
  <p>A unified view of monthly revenue across four cities, three hotel segments and three sales channels,
     with the drivers most associated with revenue and a regression-based revenue forecast. Built to answer,
     at a glance, what happened, why, and where to focus.</p>
  <div class="meta"><span class="live"></span> Last refreshed {_now} · {len(df)} monthly records</div>
</div>
""", unsafe_allow_html=True)

if f.empty:
    st.markdown("<div class='limit'><div>⚠ No records match the selected filters. Widen the filters in the sidebar.</div></div>", unsafe_allow_html=True)
    st.stop()

# ---- KPIs ----
tot=f[TARGET].sum(); avg=f[TARGET].mean(); adr=f.avg_daily_rate_usd.mean(); occ=f.occupancy_rate_pct.mean(); rating=f.online_rating.mean()
def chip(c,t): return f"<span class='chip {c}'>{t}</span>"
st.markdown("<div class='sec'><span class='bar'></span><h2>Revenue KPIs</h2>"
            f"<span class='hint'>{len(f)} of {len(df)} monthly records in view</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Total Revenue</span><span class="ico">＄</span></div>
    <div class="val">{money(tot)}</div><div class="sub">{chip('chip-neu','Sum')} across records in view</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Avg Monthly Revenue</span><span class="ico">▤</span></div>
    <div class="val">{money(avg)}</div><div class="sub">{chip('chip-neu','Per record')} mean</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Avg Daily Rate</span><span class="ico">＃</span></div>
    <div class="val">${adr:,.0f}</div><div class="sub">{chip('chip-pos','Price lever')} per night</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Avg Occupancy</span><span class="ico">％</span></div>
    <div class="val">{occ:.1f}%</div><div class="sub">{chip('chip-pos','Volume lever')} rooms sold</div></div>
  <div class="kpi"><span class="accent"></span><div class="top"><span class="label">Avg Online Rating</span><span class="ico">★</span></div>
    <div class="val">{rating:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div><div class="sub">{chip('chip-neu','Reputation')}</div></div>
</div>
""", unsafe_allow_html=True)

# ---- insights ----
city_rev=f.groupby('city')[TARGET].mean().sort_values(ascending=False)
chan_rev=f.groupby('sales_channel')[TARGET].mean().sort_values(ascending=False)
rev_corr=f[NUM+[TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
st.markdown("<div class='sec'><span class='bar'></span><h2>Key Insights</h2>"
            "<span class='hint'>auto-generated · descriptive, non-causal</span></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="ins-grid">
  <div class="ins"><div class="k">◆ Top market</div><div class="t">{city_rev.index[0]} leads revenue</div>
    <div class="d">Highest average monthly revenue at {money(city_rev.iloc[0])}; lowest is {city_rev.index[-1]} at {money(city_rev.iloc[-1])}.</div></div>
  <div class="ins"><div class="k">◆ Best channel</div><div class="t">{chan_rev.index[0]} generates the most</div>
    <div class="d">Leads at {money(chan_rev.iloc[0])} average; Direct &amp; Corporate also avoid OTA commissions.</div></div>
  <div class="ins"><div class="k">◆ Strongest driver</div><div class="t">{NUM_LABEL[rev_corr.index[0]]} moves with revenue</div>
    <div class="d">Highest correlation with revenue (r = {rev_corr.iloc[0]:.2f}), then {NUM_LABEL[rev_corr.index[1]]} (r = {rev_corr.iloc[1]:.2f}).</div></div>
</div>
""", unsafe_allow_html=True)

# ======================= TABS =======================
st.markdown("<div class='sec'><span class='bar'></span><h2>Analytics</h2>"
            "<span class='hint'>segments · relationships · distributions · trend · model</span></div>", unsafe_allow_html=True)
T1,T2,T3,T4,T5 = st.tabs(["🏙️ Cities & Segments","🔗 Relationships","📈 Distributions","🕒 Monthly Trend","🤖 Model & Forecast"])

# TAB 1
with T1:
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Average Revenue by City</div>", unsafe_allow_html=True)
        cr=f.groupby('city')[TARGET].mean().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax)
        bars=ax.bar(cr.index,cr.values,color=NAVY,width=.62,zorder=3); bars[0].set_color(ACCENT)
        ax.set_ylabel("Avg monthly revenue (USD)")
        for i,v in enumerate(cr.values): ax.text(i,v+8000,f"${v/1000:.0f}k",ha='center',fontsize=8,color=INK)
        plt.xticks(rotation=15); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{cr.index[0]}</b> leads ({money(cr.iloc[0])}); <b>{cr.index[-1]}</b> lowest ({money(cr.iloc[-1])}).</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Average Revenue by Sales Channel</div>", unsafe_allow_html=True)
        ch=f.groupby('sales_channel')[TARGET].mean().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax)
        bars=ax.bar(ch.index,ch.values,color=SLATE,width=.6,zorder=3); bars[0].set_color(TEAL)
        ax.set_ylabel("Avg monthly revenue (USD)")
        for i,v in enumerate(ch.values): ax.text(i,v+6000,f"${v/1000:.0f}k",ha='center',fontsize=8,color=INK)
        plt.xticks(rotation=12); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'><b>{ch.index[0]}</b> highest ({money(ch.iloc[0])}); <b>{ch.index[-1]}</b> lowest ({money(ch.iloc[-1])}). Direct/Corporate avoid OTA fees.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Heatmap · Average Revenue by City × Hotel Segment</div>", unsafe_allow_html=True)
    piv=f.pivot_table(index='city',columns='hotel_segment',values=TARGET,aggfunc='mean')
    piv=piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig,ax=plt.subplots(figsize=(9,4.0)); im=ax.imshow(piv.values,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]; ax.text(j,i,f"${v/1000:.0f}k",ha='center',va='center',fontsize=8,
                                       color='white' if v>piv.values.mean() else INK)
    ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label='Avg revenue (USD)'); plt.tight_layout(); fig_close(fig)
    hi=piv.stack().idxmax(); lo=piv.stack().idxmin()
    st.markdown(f"<div class='note'>Strongest combination: <b>{hi[0]} · {hi[1]}</b> ({money(piv.stack().max())}); "
                f"weakest: <b>{lo[0]} · {lo[1]}</b> ({money(piv.stack().min())}). Set strategy per city × segment.</div></div>", unsafe_allow_html=True)

# TAB 2
with T2:
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Correlation Heatmap (numeric variables)</div>", unsafe_allow_html=True)
        cols=NUM+[TARGET]; corr=f[cols].corr()
        labs=[NUM_LABEL.get(c,'Revenue') for c in cols]
        fig,ax=plt.subplots(figsize=(6.2,5.2)); im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v=corr.iloc[i,j]; ax.text(j,i,f"{v:.2f}",ha='center',va='center',fontsize=7,color='white' if abs(v)>0.55 else INK)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=40,ha='right',fontsize=7.5)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=7.5); ax.grid(False)
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Top revenue correlates: <b>{NUM_LABEL[rev_corr.index[0]]}</b> ({rev_corr.iloc[0]:.2f}) and "
                    f"<b>{NUM_LABEL[rev_corr.index[1]]}</b> ({rev_corr.iloc[1]:.2f}). Correlation ≠ causation.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Scatter · Revenue vs Average Daily Rate</div>", unsafe_allow_html=True)
        x=f.avg_daily_rate_usd.values; y=f[TARGET].values
        fig,ax=plt.subplots(figsize=(6.2,5.2)); style_ax(ax)
        ax.scatter(x,y,s=20,alpha=.45,color=ACCENT,edgecolor='white',linewidth=.4,zorder=3)
        if len(x)>2:
            b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),50); ax.plot(xs,a+b*xs,color=NAVY,lw=2,zorder=4)
        r=np.corrcoef(x,y)[0,1] if len(x)>2 else float('nan')
        ax.set_xlabel("Average daily rate (USD)"); ax.set_ylabel("Monthly revenue (USD)"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Clear upward relationship (r = <b>{r:.2f}</b>): pricing is a primary revenue lever (association, not proof).</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Scatter · Revenue vs Occupancy Rate (color = city)</div>", unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(11,4.0)); style_ax(ax)
    for i,cty in enumerate(sorted(f.city.unique())):
        sub=f[f.city==cty]; ax.scatter(sub.occupancy_rate_pct,sub[TARGET],s=22,alpha=.6,
                                        color=['#2563EB','#0E7C86','#B45309','#7C3AED'][i%4],label=cty,edgecolor='white',linewidth=.3)
    ax.set_xlabel("Occupancy rate (%)"); ax.set_ylabel("Monthly revenue (USD)"); ax.legend(frameon=False,fontsize=9); plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>Revenue rises with occupancy across all cities; higher-revenue clusters sit toward the upper-right. "
                "Filling rooms converts capacity into revenue.</div></div>", unsafe_allow_html=True)

# TAB 3
with T3:
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ctitle'>Histogram · Monthly Revenue</div>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax); ax.grid(axis='x',visible=False)
        ax.hist(f[TARGET].dropna(),bins=24,color=ACCENT,alpha=.85,edgecolor='white'); ax.axvline(f[TARGET].mean(),color=NAVY,lw=2,ls='--')
        ax.set_xlabel("Monthly revenue (USD)"); ax.set_ylabel("Frequency"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Mean {money(f[TARGET].mean())} (line). Wide spread confirms revenue varies strongly by combination.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ctitle'>Box plot · Revenue by City</div>", unsafe_allow_html=True)
        order=f.groupby('city')[TARGET].median().sort_values(ascending=False).index
        data=[f.loc[f.city==c,TARGET].values for c in order]
        fig,ax=plt.subplots(figsize=(5.4,3.4)); style_ax(ax)
        bp=ax.boxplot(data,tick_labels=list(order),patch_artist=True,medianprops=dict(color=NAVY,linewidth=2))
        for p in bp['boxes']: p.set(facecolor='#DBEAFE',edgecolor=ACCENT)
        ax.set_ylabel("Monthly revenue (USD)"); plt.xticks(rotation=12); plt.tight_layout(); fig_close(fig)
        st.markdown("<div class='note'>Median and spread by city. Higher, tighter boxes = consistently strong markets.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Box plot · Revenue by Hotel Segment</div>", unsafe_allow_html=True)
    order=f.groupby('hotel_segment')[TARGET].median().sort_values(ascending=False).index
    data=[f.loc[f.hotel_segment==sgmt,TARGET].values for sgmt in order]
    fig,ax=plt.subplots(figsize=(11,3.4)); style_ax(ax)
    bp=ax.boxplot(data,tick_labels=list(order),patch_artist=True,vert=True,medianprops=dict(color=NAVY,linewidth=2))
    for p in bp['boxes']: p.set(facecolor='#DBEAFE',edgecolor=ACCENT)
    ax.set_ylabel("Monthly revenue (USD)"); plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>Revenue distribution by segment (Business / Conference / Leisure). Segment mix shapes revenue potential alongside city.</div></div>", unsafe_allow_html=True)

# TAB 4
with T4:
    st.markdown("<div class='card'><div class='ctitle'>Monthly Revenue Trend (total per month)</div>", unsafe_allow_html=True)
    mon=f.groupby('month')[TARGET].sum()
    fig,ax=plt.subplots(figsize=(11,3.8)); style_ax(ax)
    ax.fill_between(range(len(mon)),mon.values,color=ACCENT,alpha=.08,zorder=2)
    ax.plot(range(len(mon)),mon.values,color=ACCENT,lw=2.4,zorder=3)
    ax.scatter(range(len(mon)),mon.values,color=ACCENT,s=26,zorder=4,edgecolor='white',linewidth=1.2)
    imax=int(np.argmax(mon.values)); ax.scatter([imax],[mon.values[imax]],color=NAVY,s=70,zorder=5,edgecolor='white',linewidth=1.4)
    ax.set_xticks(range(len(mon))); ax.set_xticklabels(mon.index,rotation=45,ha='right',fontsize=8)
    ax.set_ylabel("Total monthly revenue (USD)"); plt.tight_layout(); fig_close(fig)
    st.markdown(f"<div class='note'>Peak month: <b>{mon.idxmax()}</b> ({money(mon.max())}); lowest: <b>{mon.idxmin()}</b> ({money(mon.min())}). "
                f"Use seasonality to plan rates, staffing and marketing.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='ctitle'>Average Monthly Revenue by Month and City</div>", unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(11,3.8)); style_ax(ax)
    for i,cty in enumerate(sorted(f.city.unique())):
        sub=f[f.city==cty].groupby('month')[TARGET].mean()
        ax.plot(range(len(sub)),sub.values,marker='o',ms=4,lw=1.8,label=cty,
                color=['#2563EB','#0E7C86','#B45309','#7C3AED'][i%4])
    ax.set_xticks(range(len(mon))); ax.set_xticklabels(mon.index,rotation=45,ha='right',fontsize=8)
    ax.set_ylabel("Avg monthly revenue (USD)"); ax.legend(frameon=False,fontsize=9,ncol=4); plt.tight_layout(); fig_close(fig)
    st.markdown("<div class='note'>Each city's monthly path. Diverging lines show markets peak at different times — tailor campaigns by city and month.</div></div>", unsafe_allow_html=True)

# TAB 5
with T5:
    st.markdown("<div class='sec' style='margin-top:6px'><span class='bar'></span><h2>Revenue Forecast · Scenario Center</h2>"
                "<span class='hint'>defaults reproduce the workshop scenario · ≈ $707,029</span></div>", unsafe_allow_html=True)
    left,right=st.columns([1.45,1],gap="large")
    with left:
        st.markdown("<div class='card' style='padding-bottom:18px'><div class='ctitle'>Scenario inputs · 8 predictors</div>", unsafe_allow_html=True)
        cc1,cc2=st.columns(2)
        with cc1:
            rooms=st.number_input("Available rooms",50,500,160)
            adr_i=st.number_input("Average daily rate (USD)",50.0,500.0,145.0)
            occ_i=st.number_input("Occupancy rate (%)",10.0,100.0,74.0)
            mkt_i=st.number_input("Marketing spend (USD)",0.0,200000.0,33000.0,step=1000.0)
        with cc2:
            rat_i=st.number_input("Online rating",1.0,5.0,4.35,step=0.05)
            cpi_i=st.number_input("Competitor price index",0.5,2.0,1.02,step=0.01)
            bwd_i=st.number_input("Booking window (days)",1.0,120.0,24.0)
            loy_i=st.number_input("Loyalty members",0,10000,1250)
        st.button("◆  Generate Forecast", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    scenario=pd.DataFrame([{'available_rooms':rooms,'avg_daily_rate_usd':adr_i,'occupancy_rate_pct':occ_i,
        'marketing_spend_usd':mkt_i,'online_rating':rat_i,'competitor_price_index':cpi_i,
        'booking_window_days':bwd_i,'loyalty_members':loy_i}])
    pred=float(model.predict(scenario[NUM])[0])
    lvl = "Above average" if pred>=avg else "Below average"
    lc = "#34D399" if pred>=avg else "#FBBF24"
    with right:
        st.markdown(f"""
        <div class="fc-wrap"><div class="fc-value">
          <div class="lab">Forecasted Monthly Revenue</div>
          <div class="big">{money(pred)}</div>
          <div class="pill"><span style="width:8px;height:8px;border-radius:50%;background:{lc};display:inline-block"></span>
            {lvl} vs portfolio mean</div>
          <div style="margin-top:14px;font-size:12.5px;color:#9DB4FF;line-height:1.5">
            In-sample model · R² {metrics['r2']:.3f} · typical error ≈ {money(metrics['mae'])}.
            Planning estimate, not a guaranteed result.</div>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec' style='margin-top:18px'><span class='bar'></span><h2>Model Diagnostics</h2></div>", unsafe_allow_html=True)
    g1,g2,g3=st.columns(3,gap="large")
    with g1:
        st.markdown("<div class='card'><div class='ctitle'>Predicted vs Actual (in-sample)</div>", unsafe_allow_html=True)
        ya=df[TARGET].values; yp=y_pred_full
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax)
        ax.scatter(ya,yp,s=14,alpha=.4,color=ACCENT,edgecolor='white',linewidth=.3,zorder=3)
        lim=[ya.min(),ya.max()]; ax.plot(lim,lim,color=NAVY,ls='--',lw=1.5)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>R² {metrics['r2']:.3f}. Points hug the diagonal — strong in-sample fit.</div></div>", unsafe_allow_html=True)
    with g2:
        st.markdown("<div class='card'><div class='ctitle'>Residual Histogram</div>", unsafe_allow_html=True)
        res=df[TARGET].values-y_pred_full
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax); ax.grid(axis='x',visible=False)
        ax.hist(res,bins=22,color=NAVY,alpha=.85,edgecolor='white'); ax.axvline(0,color=ACCENT,lw=2)
        ax.set_xlabel("Residual (actual − predicted)"); ax.set_ylabel("Frequency"); plt.tight_layout(); fig_close(fig)
        st.markdown(f"<div class='note'>Centered near 0 (MAE ≈ {money(metrics['mae'])}) → no systematic bias.</div></div>", unsafe_allow_html=True)
    with g3:
        st.markdown("<div class='card'><div class='ctitle'>Driver Importance (permutation)</div>", unsafe_allow_html=True)
        idf=driver_importance(model,df).head(8).copy()
        idf['lab']=idf['Feature'].map(NUM_LABEL); idf=idf.sort_values('imp')
        fig,ax=plt.subplots(figsize=(4.2,3.4)); style_ax(ax); ax.grid(axis='y',visible=False); ax.grid(axis='x',visible=True)
        ax.barh(idf['lab'],idf['imp'],color=ACCENT,zorder=3); ax.set_xlabel("Drop in R² when shuffled")
        plt.tight_layout(); fig_close(fig)
        top=driver_importance(model,df).iloc[0]
        st.markdown(f"<div class='note'>Top driver: <b>{NUM_LABEL[top['Feature']]}</b>. Diagnostic of model reliance, not causal proof.</div></div>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{SLATE};font-size:13.5px;line-height:1.6;margin:16px 2px 12px'>"
        f"The multiple linear regression predicts <b style='color:{INK}'>monthly revenue</b> from eight operating, pricing, "
        f"customer and competitive variables, fit on the complete dataset. It explains about "
        f"<b style='color:{INK}'>{metrics['r2']*100:.0f}%</b> of revenue variation (R² {metrics['r2']:.3f}), with a typical error of "
        f"<b style='color:{INK}'>{money(metrics['mae'])}</b> (MAE) and {money(metrics['rmse'])} (RMSE). The strongest revenue levers are "
        f"<b style='color:{INK}'>average daily rate</b> and <b style='color:{INK}'>occupancy</b>.</p>",
        unsafe_allow_html=True)
    st.markdown("<div class='limit'><div>⚠ <b>Limitation.</b> Metrics are <b>in-sample</b> — the model was fit on the complete "
        "dataset with <b>no train/test split</b>, so they describe fit to history, not validated future accuracy. Coefficients are "
        "associations within the model, not proof of causation. Treat forecasts as planning estimates.</div></div>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:12px;margin-top:26px;"
            f"padding-top:16px;border-top:1px solid {LINE}'>"
            "Hotel Revenue Analytics · Individual BI Workshop &nbsp;·&nbsp; metrics are in-sample</div>", unsafe_allow_html=True)
