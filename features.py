"""Feature engineering.

Features are organised in named groups so the training step can run an ablation
study (drop one group at a time and see how much cross-validated R^2 falls).
"""
import numpy as np
import pandas as pd

from . import config as cfg

TM_COLS = ["tm_min_all", "tm_min_pl", "tm_apps", "tm_starts", "tm_goals", "tm_assists",
           "tm_yellow", "tm_red", "contract_years_left", "intl_caps", "intl_goals"]

FEATURE_GROUPS = {
    # Value falls steeply after ~28; hinge terms let the slope change. Keepers peak
    # later than outfield players, hence the goalkeeper-by-age terms.
    "age": ["age", "age_sq", "old", "young", "old_x_contrib", "gk_x_age", "gk_x_old"],
    "league_stats": ["l_goals", "l_assists", "l_np_goals", "l_played", "l_log_contrib"],
    "club_strength": ["club_pos", "club_pts", "club_gd"],
    # Minutes/starts separate a starter from a backup at the same club.
    "season_activity": ["log_min", "tm_starts", "contrib_per90", "tm_yellow",
                        "gk_x_log_min", "def_x_log_min"],
    "contract_caps": ["contract_years_left", "contract_missing", "log_caps"],
    "position": ["pos_Goalkeeper", "pos_Defence", "pos_Midfield", "pos_Offence"],
    "world_cup": ["wc_goals", "wc_assists", "wc_contrib", "went_wc"],
    # Off by default: the market values predate the 2026/27 season.
    "current_season": ["c_goals", "c_assists", "c_played"],
}


def load_stats() -> pd.DataFrame:
    """stats.csv (football-data.org) joined with tm_features.csv (Transfermarkt)."""
    if not cfg.STATS_CSV.exists():
        raise SystemExit(f"{cfg.STATS_CSV} not found. Run: python -m plvalue fetch")
    if not cfg.TM_FEATURES_CSV.exists():
        raise SystemExit(f"{cfg.TM_FEATURES_CSV} not found. Run: python -m plvalue build")
    stats = pd.read_csv(cfg.STATS_CSV)
    feats = pd.read_csv(cfg.TM_FEATURES_CSV).drop_duplicates("name")
    stats = stats.merge(feats, on="name", how="left")
    for c in TM_COLS:
        if c != "contract_years_left":
            stats[c] = stats[c].fillna(0)  # no record = no minutes / caps
    return stats


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    for pos in ("Goalkeeper", "Defence", "Midfield", "Offence"):
        X[f"pos_{pos}"] = df["position"].str.contains(pos, case=False, na=False).astype(int)

    age = df["age"].fillna(df["age"].median())
    X["age"], X["age_sq"] = age, age ** 2
    X["old"] = (age - 28).clip(lower=0)
    X["young"] = (23 - age).clip(lower=0)
    X["gk_x_age"] = X["pos_Goalkeeper"] * age
    X["gk_x_old"] = X["pos_Goalkeeper"] * X["old"]

    X["l_goals"], X["l_assists"], X["l_played"] = df["l_goals"], df["l_assists"], df["l_played"]
    X["l_np_goals"] = df["l_goals"] - df["l_pens"]
    X["l_log_contrib"] = np.log1p(df["l_goals"] + df["l_assists"])
    X["old_x_contrib"] = X["old"] * X["l_log_contrib"]
    X["c_goals"], X["c_assists"], X["c_played"] = df["c_goals"], df["c_assists"], df["c_played"]

    for c in ("club_pos", "club_pts", "club_gd"):
        X[c] = df[c]

    mins = df["tm_min_all"]
    X["log_min"] = np.log1p(mins)
    X["tm_starts"] = df["tm_starts"]
    # Floor at 900 minutes so a short hot run can't dominate; cap at 1.5 G+A per 90.
    X["contrib_per90"] = (90 * (df["tm_goals"] + df["tm_assists"]) / mins.clip(lower=900)).clip(upper=1.5)
    X["tm_yellow"] = df["tm_yellow"]
    X["gk_x_log_min"] = X["pos_Goalkeeper"] * X["log_min"]
    X["def_x_log_min"] = X["pos_Defence"] * X["log_min"]

    cy = df["contract_years_left"]
    X["contract_missing"] = cy.isna().astype(int)
    X["contract_years_left"] = cy.fillna(1.5)
    X["log_caps"] = np.log1p(df["intl_caps"])

    X["wc_goals"], X["wc_assists"] = df["wc_goals"], df["wc_assists"]
    X["wc_contrib"] = df["wc_goals"] + df["wc_assists"]
    X["went_wc"] = df["went_wc"]
    return X
