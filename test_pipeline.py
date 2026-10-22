import json

import numpy as np
import pandas as pd

from plvalue import config as cfg
from plvalue import model, transfermarkt


def test_build_matches_every_synthetic_player(workspace):
    transfermarkt.build()
    values = pd.read_csv(cfg.VALUES_CSV)
    feats = pd.read_csv(cfg.TM_FEATURES_CSV)
    assert len(values) == 250 and len(feats) == 250
    assert (values["value_m"] > 0).all()
    assert {"tm_min_all", "tm_starts", "contract_years_left", "intl_caps"} <= set(feats.columns)


def test_train_predict_and_reports(workspace):
    transfermarkt.build()
    model.train("linear")

    metrics = json.loads((cfg.REPORTS_DIR / "metrics.json").read_text())
    assert metrics["n_players"] == 250
    assert metrics["cv_r2"] > 0.5                      # synthetic signal is strong
    assert "world_cup" in metrics["ablation"]
    assert (cfg.REPORTS_DIR / "pred_vs_actual.png").exists()
    assert (cfg.REPORTS_DIR / "results.md").read_text().startswith("Trained on")

    bundle = model.load_bundle()
    rows = pd.read_csv(cfg.STATS_CSV).head(5).merge(
        pd.read_csv(cfg.TM_FEATURES_CSV), on="name", how="left")
    res = model.predict_frame(rows, bundle)
    assert (res["low_m"] <= res["pred_m"]).all() and (res["pred_m"] <= res["high_m"]).all()
    assert np.isfinite(res[["pred_m", "low_m", "high_m"]].to_numpy()).all()


def test_out_of_fold_never_scores_a_player_with_a_model_that_saw_them(workspace):
    transfermarkt.build()
    df = model.load_training_frame()
    from plvalue.features import build_features
    X, y = build_features(df), np.log1p(df["value_m"])
    _, cols = model.select_columns()
    out, resid = model.out_of_fold(df, X[cols], y, "linear")
    assert len(out) == len(df) and len(resid) == len(df)
    # in-sample fit is optimistic; out-of-fold error must be at least as large
    fitted = model.make_model("linear").fit(X[cols], y).predict(X[cols])
    assert np.abs(resid).mean() >= np.abs(y.to_numpy() - fitted).mean()


def test_update_readme_replaces_between_markers(workspace, tmp_path, monkeypatch):
    transfermarkt.build()
    model.train("linear")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    (tmp_path / "README.md").write_text("# T\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nend\n")
    model.update_readme()
    text = (tmp_path / "README.md").read_text()
    assert "\nold\n" not in text and "Cross-validated R²" in text and text.rstrip().endswith("end")
