from datetime import date

import numpy as np
import pandas as pd

from plvalue.features import FEATURE_GROUPS, build_features
from plvalue.transfermarkt import name_score
from plvalue.utils import age_from_dob, norm


def test_norm_strips_accents_and_case():
    assert norm("  Martin  Ødegaard ") == "martin degaard" or norm("Ødegaard").endswith("degaard")
    assert norm("Kylian MBAPPÉ") == "kylian mbappe"


def test_age_from_dob():
    assert age_from_dob("2000-01-01", today=date(2026, 1, 1)) == 26.0
    assert np.isnan(age_from_dob("not a date"))


def test_name_score_handles_short_names():
    assert name_score("thiago", "igor thiago") >= 0.9      # token subset
    assert name_score("mickey van de ven", "micky van de ven") > 0.9
    assert name_score("alice smith", "zhang wei") < 0.5


def _row(**over):
    base = dict(position="Midfield", age=30.0, l_goals=5, l_assists=8, l_pens=1, l_played=30,
                c_goals=0, c_assists=0, c_played=0, club_pos=3, club_pts=70, club_gd=20,
                tm_min_all=2500, tm_starts=28, tm_goals=6, tm_assists=9, tm_yellow=4,
                contract_years_left=2.0, intl_caps=40, wc_goals=0, wc_assists=0, went_wc=1)
    base.update(over)
    return pd.DataFrame([base])


def test_every_group_column_is_built():
    X = build_features(_row())
    needed = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert set(needed) <= set(X.columns)
    assert not X[needed].isna().any().any()


def test_age_hinge_terms():
    assert build_features(_row(age=33.0))["old"].iloc[0] == 5.0
    assert build_features(_row(age=20.0))["young"].iloc[0] == 3.0
    assert build_features(_row(age=25.0))[["old", "young"]].iloc[0].sum() == 0


def test_goalkeeper_interactions_only_fire_for_keepers():
    gk = build_features(_row(position="Goalkeeper", age=34.0))
    out = build_features(_row(position="Midfield", age=34.0))
    assert gk["gk_x_age"].iloc[0] == 34.0 and out["gk_x_age"].iloc[0] == 0


def test_per90_is_stabilised_and_capped():
    # 1 minute, 5 goals: floor on minutes (900) and cap at 1.5 stop it exploding
    X = build_features(_row(tm_min_all=1, tm_goals=5, tm_assists=5))
    assert X["contrib_per90"].iloc[0] <= 1.5


def test_missing_contract_is_flagged_not_dropped():
    X = build_features(_row(contract_years_left=np.nan))
    assert X["contract_missing"].iloc[0] == 1 and X["contract_years_left"].iloc[0] == 1.5
