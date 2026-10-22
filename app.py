"""Streamlit demo:  streamlit run app.py   (run `python -m plvalue train` first)."""
import json

import pandas as pd
import streamlit as st

from plvalue import config as cfg
from plvalue.features import load_stats
from plvalue.model import load_bundle, predict_frame

st.set_page_config(page_title="PL Player Value Predictor", page_icon="⚽", layout="wide")


@st.cache_resource
def load_all():
    bundle, stats = load_bundle(), load_stats()
    oof_path, metrics_path = cfg.REPORTS_DIR / "oof_predictions.csv", cfg.REPORTS_DIR / "metrics.json"
    oof = pd.read_csv(oof_path) if oof_path.exists() else None
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return bundle, stats, oof, metrics


st.title("⚽ Premier League player value predictor")
st.caption("Predicts a player's Transfermarkt market value from season stats, minutes, age, "
           "club strength, contract length and caps.")

if not cfg.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m plvalue train` first.")
    st.stop()

bundle, stats, oof, metrics = load_all()
tab_lookup, tab_perf, tab_misses = st.tabs(["Player lookup", "Model performance", "Biggest misses"])

with tab_lookup:
    name = st.selectbox("Choose a player", sorted(stats["name"].unique()))
    row = stats[stats["name"] == name].head(1)
    res = predict_frame(row, bundle).iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted value", f"€{res['pred_m']:.1f}m")
    c2.metric("Likely range (80%)", f"€{res['low_m']:.0f}m - €{res['high_m']:.0f}m")
    actual = None
    if oof is not None and name in set(oof["name"]):
        actual = float(oof.loc[oof["name"] == name, "actual_m"].iloc[0])
    c3.metric("Transfermarkt value", f"€{actual:.1f}m" if actual is not None else "n/a",
              delta=f"{res['pred_m'] - actual:+.1f}m model vs actual" if actual is not None else None,
              delta_color="off")

    r = row.iloc[0]
    st.write(f"**{r['team']}** · {r['position']} · age {r['age']} · "
             f"{int(r['tm_min_all']):,} minutes in 2025/26 · "
             f"{int(r['tm_goals'])} goals, {int(r['tm_assists'])} assists")
    if int(r["went_wc"]):
        st.write("Selected in a 2026 World Cup squad.")

with tab_perf:
    if metrics:
        lo, hi = metrics["interval_80_pct"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cross-validated R²", f"{metrics['cv_r2']:.2f}")
        c2.metric("Mean absolute error", f"€{metrics['mae_m']:.1f}m")
        c3.metric("Median error", f"{metrics['median_abs_pct_err']:.0f}%")
        st.subheader("Which features matter? (ablation study)")
        st.dataframe(pd.DataFrame(metrics["ablation"]).T.rename(
            columns={"r2": "R² without group", "delta": "Change in R²"}).round(3))
    img = cfg.REPORTS_DIR / "pred_vs_actual.png"
    if img.exists():
        st.image(str(img), caption="Out-of-fold predictions vs Transfermarkt values (log scale)")

with tab_misses:
    if oof is None:
        st.info("Run `python -m plvalue train` to generate the error report.")
    else:
        cols = ["name", "team", "age", "actual_m", "pred_m", "err_m"]
        st.subheader("Most overvalued by the model")
        st.dataframe(oof.nlargest(15, "err_m")[cols].round(1), hide_index=True)
        st.subheader("Most undervalued by the model")
        st.dataframe(oof.nsmallest(15, "err_m")[cols].round(1), hide_index=True)
