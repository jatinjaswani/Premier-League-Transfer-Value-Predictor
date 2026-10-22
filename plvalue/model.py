"""Training, evaluation (out-of-fold), ablation study, and prediction."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config as cfg  # noqa: E402
from .features import FEATURE_GROUPS, build_features, load_stats  # noqa: E402
from .utils import norm  # noqa: E402

AGE_BANDS = [0, 21, 24, 27, 30, 33, 60]
AGE_LABELS = ["<=21", "22-24", "25-27", "28-30", "31-33", "34+"]


def make_model(name: str):
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    if name == "gbm":
        return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=cfg.SEED)
    raise SystemExit(f"Unknown model '{name}' (choose linear, ridge or gbm)")


def select_columns(use_current: bool = False):
    groups = {g: c for g, c in FEATURE_GROUPS.items() if use_current or g != "current_season"}
    return groups, [c for cs in groups.values() for c in cs]


def _cv():
    return KFold(5, shuffle=True, random_state=cfg.SEED)


def cv_r2(model_name, X, y) -> float:
    return float(cross_val_score(make_model(model_name), X, y, cv=_cv(), scoring="r2").mean())


def load_training_frame() -> pd.DataFrame:
    stats = load_stats()
    if not cfg.VALUES_CSV.exists():
        raise SystemExit(f"{cfg.VALUES_CSV} not found. Run: python -m plvalue build")
    vals = pd.read_csv(cfg.VALUES_CSV)
    stats["key"], vals["key"] = stats["name"].map(norm), vals["name"].map(norm)
    df = stats.merge(vals[["key", "value_m"]], on="key").query("value_m > 0").reset_index(drop=True)
    print(f"Matched {len(df)} of {len(stats)} players to a market value")
    return df


def out_of_fold(df, X, y, model_name):
    """Predict every player with a model that never saw them. Returns (table, log residuals)."""
    oof = cross_val_predict(make_model(model_name), X, y, cv=_cv())
    out = df[["name", "team", "position", "age"]].copy()
    out["actual_m"], out["pred_m"] = np.expm1(y.to_numpy()), np.expm1(oof)
    out["pct_err"] = (out["pred_m"] / out["actual_m"] - 1) * 100
    out["err_m"] = out["pred_m"] - out["actual_m"]
    return out, y.to_numpy() - oof


def age_band_table(out: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(out["age"], AGE_BANDS, labels=AGE_LABELS)
    return out.groupby(bands, observed=True)["pct_err"].agg(["median", "count"]).round(1)


def print_report(out, top=0):
    mae = (out["pred_m"] - out["actual_m"]).abs().mean()
    print(f"\nOut-of-fold MAE: {mae:.1f}m | median absolute error: {out.pct_err.abs().median():.0f}%")
    print("\nMedian % error by age band (positive = model overvalues):")
    print(age_band_table(out))
    if top:
        cols = ["name", "team", "age", "actual_m", "pred_m", "err_m"]
        print(f"\nMost OVERVALUED by the model, ranked by euros (top {top}):")
        print(out.nlargest(top, "err_m")[cols].round(1).to_string(index=False))
        print(f"\nMost UNDERVALUED by the model, ranked by euros (top {top}):")
        print(out.nsmallest(top, "err_m")[cols].round(1).to_string(index=False))


def plot_pred_vs_actual(out: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(out["actual_m"], out["pred_m"].clip(lower=0.05), s=14, alpha=0.55)
    lo, hi = 0.1, max(out["actual_m"].max(), out["pred_m"].max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color="gray")
    for _, r in out.reindex(out["err_m"].abs().nlargest(6).index).iterrows():
        ax.annotate(r["name"].split()[-1], (r["actual_m"], max(r["pred_m"], 0.05)),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Transfermarkt market value (EUR m)")
    ax.set_ylabel("Model prediction (EUR m, out-of-fold)")
    ax.set_title("Predicted vs actual market value")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _results_markdown(m: dict) -> str:
    lo, hi = m["interval_80_pct"]
    lines = [
        f"Trained on **{m['n_players']} Premier League players** (model: `{m['model']}`, "
        "5-fold cross-validation; every player is scored by a model that never saw them).", "",
        "| Metric | Value |", "|---|---|",
        f"| Cross-validated R² (log value) | {m['cv_r2']:.3f} |",
        f"| Mean absolute error | €{m['mae_m']:.1f}m |",
        f"| Median absolute error | {m['median_abs_pct_err']:.0f}% |",
        f"| 80% prediction interval | {lo:+.0f}% to {hi:+.0f}% of the point estimate |", "",
        "**Model comparison** (all features, cross-validated R²)", "",
        "| Model | R² |", "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in m["model_comparison"].items()], "",
        "**Ablation study** (drop one feature group, retrain, measure the change in R²)", "",
        "| Feature group removed | R² | Change |", "|---|---|---|",
        *[f"| {k} | {v['r2']:.3f} | {v['delta']:+.3f} |" for k, v in m["ablation"].items()],
    ]
    return "\n".join(lines) + "\n"


def train(model_name="linear", use_current=False):
    df = load_training_frame()
    X, y = build_features(df), np.log1p(df["value_m"])
    groups, cols = select_columns(use_current)

    print("\n5-fold CV R^2 (log value), all features:")
    comparison = {}
    for nm in ("linear", "ridge", "gbm"):
        comparison[nm] = cv_r2(nm, X[cols], y)
        print(f"  {nm:7} {comparison[nm]:.3f}")

    full = comparison[model_name]
    print(f"\nAblation ({model_name}): R^2 without each feature group (full model = {full:.3f})")
    ablation = {}
    for g, gcols in groups.items():
        r2 = cv_r2(model_name, X[[c for c in cols if c not in gcols]], y)
        ablation[g] = {"r2": r2, "delta": r2 - full}
        print(f"  without {g:16} {r2:.3f}  ({r2 - full:+.3f})")

    out, resid = out_of_fold(df, X[cols], y, model_name)
    print_report(out)
    q10, q90 = (float(q) for q in np.quantile(resid, [0.1, 0.9]))

    model = make_model(model_name).fit(X[cols], y)
    if model_name == "linear":
        print("\nWeights:\n", pd.Series(model.coef_, index=cols).sort_values().round(3))

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols, "model_name": model_name,
                 "interval": (q10, q90)}, cfg.MODEL_PATH)

    metrics = {
        "n_players": int(len(df)), "model": model_name, "cv_r2": full,
        "mae_m": float((out["pred_m"] - out["actual_m"]).abs().mean()),
        "median_abs_pct_err": float(out["pct_err"].abs().median()),
        "interval_80_pct": [float((np.exp(q10) - 1) * 100), float((np.exp(q90) - 1) * 100)],
        "model_comparison": comparison, "ablation": ablation,
        "age_band_median_pct_err": age_band_table(out)["median"].to_dict(),
    }
    (cfg.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (cfg.REPORTS_DIR / "results.md").write_text(_results_markdown(metrics))
    out.round(2).to_csv(cfg.REPORTS_DIR / "oof_predictions.csv", index=False)
    plot_pred_vs_actual(out, cfg.REPORTS_DIR / "pred_vs_actual.png")
    print(f"\nSaved model -> {cfg.MODEL_PATH}\nSaved metrics, results.md, oof_predictions.csv and "
          f"pred_vs_actual.png -> {cfg.REPORTS_DIR}")


def load_bundle():
    if not cfg.MODEL_PATH.exists():
        raise SystemExit(f"{cfg.MODEL_PATH} not found. Run: python -m plvalue train")
    return joblib.load(cfg.MODEL_PATH)


def errors(top=12):
    b = load_bundle()
    df = load_training_frame()
    X, y = build_features(df), np.log1p(df["value_m"])
    out, _ = out_of_fold(df, X[b["columns"]], y, b["model_name"])
    print_report(out, top=top)


def predict_frame(rows: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """Point estimate plus an 80% range (from out-of-fold residuals) for each row."""
    b = bundle or load_bundle()
    log_pred = b["model"].predict(build_features(rows)[b["columns"]])
    q10, q90 = b["interval"]
    res = rows[["name", "team", "position", "age"]].copy()
    res["pred_m"] = np.expm1(log_pred).clip(0)
    res["low_m"] = np.expm1(log_pred + q10).clip(0)
    res["high_m"] = np.expm1(log_pred + q90).clip(0)
    return res


def predict(name: str):
    stats = load_stats()
    rows = stats[stats["name"].map(norm).str.contains(norm(name), regex=False)]
    if rows.empty:
        return print("Player not found")
    for _, r in predict_frame(rows).iterrows():
        print(f"{r['name']} ({r['team']}, age {r['age']}): ~ EUR {r['pred_m']:.1f}m "
              f"(80% range {r['low_m']:.1f}-{r['high_m']:.1f}m)")


def update_readme(readme="README.md"):
    """Paste reports/results.md between the RESULTS markers in the README."""
    path = cfg.ROOT / readme
    snippet = (cfg.REPORTS_DIR / "results.md").read_text()
    text = path.read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        raise SystemExit("README.md is missing the RESULTS:START / RESULTS:END markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{snippet}{end}{tail}")
    print(f"Updated {path}")
