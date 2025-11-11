
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import requests

DATA = Path("data/processed")
PLOTS = DATA / "plots"

st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("⚡ Energy Demand Forecasting — Demo Dashboard")

API_BASE = "https://energy-forecasting-03iz.onrender.com"

# User selects forecast horizon
horizon = st.selectbox("Forecast horizon (hours)", [24, 168], index=0)

try:
    res = requests.get(f"{API_BASE}/forecast", params={"horizon": horizon}, timeout=8)

    if res.ok:
        val = res.json().get("forecast_mw", None)
        if val is not None:
            st.success(f"✅ API forecast (t+{horizon}h): {val:,.1f} MW")
        else:
            st.warning("⚠️ No forecast value in API response.")
    else:
        st.error(f"❌ API error {res.status_code}: {res.text}")

except Exception as e:
    st.warning(f"🌐 Could not reach API: {e}")
    st.info("Ensure your Render API is live at https://energy-forecasting-03iz.onrender.com")

horizon = st.selectbox("Forecast horizon", [24, 168], index=0)

# Load predictions & metrics
oof_path = DATA / f"oof_h{horizon}.csv"
met_path = DATA / f"metrics_h{horizon}.csv"
fi_path  = DATA / f"feature_importance_h{horizon}.csv"

if not oof_path.exists():
    st.warning(f"Run backtest to generate {oof_path}")
    st.stop()

oof = pd.read_csv(oof_path)
y = oof["y_true"].values
p = oof["y_pred_lgbm"].values
b = oof["y_pred_baseline"].values

def mape(y, p): return np.mean(np.abs((y - p) / np.clip(np.abs(y), 1e-6, None))) * 100
def wape(y, p): return np.sum(np.abs(y - p))/np.sum(np.abs(y))*100
def rmse(y, p): return np.sqrt(np.mean((y - p)**2))

col1, col2, col3 = st.columns(3)
col1.metric("MAPE (LGBM)", f"{mape(y,p):.2f}%")
col2.metric("WAPE (LGBM)", f"{wape(y,p):.2f}%")
col3.metric("RMSE (LGBM)", f"{rmse(y,p):.1f}")

# Interactive time range
st.subheader("Actual vs Predicted")
n = len(y)
start = st.slider("Start index", 0, max(0, n-500), max(0, n-300))
end   = st.slider("End index", start+50, n, n)

fig = go.Figure()
fig.add_trace(go.Scatter(y=y[start:end], name="Actual"))
fig.add_trace(go.Scatter(y=p[start:end], name="LGBM"))
if not np.isnan(b).all():
    fig.add_trace(go.Scatter(y=b[start:end], name="Baseline", opacity=0.6))
fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Time index", yaxis_title="Load (MW)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance (Top 20)")
if fi_path.exists():
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(20)
    st.dataframe(fi)
    bar = go.Figure(go.Bar(x=fi["importance"], y=fi.iloc[:,0], orientation="h"))
    bar.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, use_container_width=True)
else:
    st.info("Run backtest to save feature importance CSV.")

st.subheader("Saved Analysis Plots")
for png in ["oof_h24.png","rolling_mae_h24.png","feature_importance_h24.png","shap_summary_h24.png","shap_bar_h24.png"]:
    p = PLOTS / png
    if p.exists():
        st.image(str(p), caption=png)
