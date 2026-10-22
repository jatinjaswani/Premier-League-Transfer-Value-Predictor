"""Load the Transfermarkt Kaggle files, derive season features, match players to stats.csv.

Matching order for each football-data player:
  1. exact normalised name (closest date of birth wins if several share it)
  2. same date of birth (+/- 1 day) and a similar name
  3. no dob available: similar name and age within 0.15 years
"""
import difflib
from datetime import date

import pandas as pd

from . import config as cfg
from .utils import norm

FEATURE_COLS = ["tm_min_all", "tm_min_pl", "tm_apps", "tm_starts", "tm_goals", "tm_assists",
                "tm_yellow", "tm_red", "contract_years_left", "intl_caps", "intl_goals"]
# Matches that only agreed on birth date with a weak name match; unverified, so skipped.
EXCLUDE = {"Tyler Fletcher", "Promise Akinpelu", "Jair Paula"}
DAY = pd.Timedelta(days=1)


def _require_raw(*names):
    missing = [n for n in names if not (cfg.RAW_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {cfg.RAW_DIR}. Download the Kaggle dataset "
            "'davidcariboo/player-scores' and copy those files into that folder.")


def load_transfermarkt() -> pd.DataFrame:
    """One row per recently-active player: latest valuation + 2025/26 activity."""
    _require_raw("players.csv", "player_valuations.csv", "appearances.csv")
    players = pd.read_csv(cfg.RAW_DIR / "players.csv",
                          usecols=["player_id", "name", "date_of_birth", "last_season",
                                   "contract_expiration_date", "international_caps",
                                   "international_goals"])
    vals = pd.read_csv(cfg.RAW_DIR / "player_valuations.csv",
                       usecols=["player_id", "date", "market_value_in_eur"])
    latest = vals.sort_values("date").groupby("player_id").tail(1)
    tm = players.merge(latest, on="player_id")
    tm = tm[(tm["last_season"] >= 2025) & (tm["market_value_in_eur"] > 0)].copy()
    tm["key"] = tm["name"].map(norm)
    tm["dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")
    tm["age"] = (pd.Timestamp(date.today()) - tm["dob"]).dt.days / 365.25

    app = pd.read_csv(cfg.RAW_DIR / "appearances.csv",
                      usecols=["player_id", "date", "competition_id", "yellow_cards",
                               "red_cards", "goals", "assists", "minutes_played"])
    app = app[(app["date"] >= cfg.SEASON_START) & (app["date"] <= cfg.SEASON_END)].copy()
    app["start"] = (app["minutes_played"] >= 60).astype(int)  # 60+ minutes ~ a start
    app["pl_min"] = app["minutes_played"].where(app["competition_id"] == "GB1", 0)
    agg = app.groupby("player_id").agg(
        tm_min_all=("minutes_played", "sum"), tm_min_pl=("pl_min", "sum"),
        tm_apps=("minutes_played", "size"), tm_starts=("start", "sum"),
        tm_goals=("goals", "sum"), tm_assists=("assists", "sum"),
        tm_yellow=("yellow_cards", "sum"), tm_red=("red_cards", "sum"),
    ).reset_index()
    tm = tm.merge(agg, on="player_id", how="left")
    for c in agg.columns[1:]:
        tm[c] = tm[c].fillna(0)  # no appearances recorded = 0 minutes
    tm["contract_years_left"] = (
        (pd.to_datetime(tm["contract_expiration_date"], errors="coerce")
         - pd.Timestamp(cfg.SEASON_END)).dt.days / 365.25).clip(0, 6)  # NaN if unknown
    tm["intl_caps"] = tm["international_caps"].fillna(0)
    tm["intl_goals"] = tm["international_goals"].fillna(0)
    return tm


def name_score(a: str, b: str) -> float:
    r = difflib.SequenceMatcher(None, a, b).ratio()
    sa, sb = set(a.split()), set(b.split())
    if sa <= sb or sb <= sa:  # e.g. "thiago" inside "igor thiago"
        r = max(r, 0.9)
    return r


def match_players(stats: pd.DataFrame, tm: pd.DataFrame):
    """Return (matched rows, fuzzy matches, dob warnings, unmatched names)."""
    out, fuzzy, unmatched, warn = [], [], [], []
    today = pd.Timestamp(date.today())

    for _, s in stats.iterrows():
        has_dob = pd.notna(s["dob"])
        best = None

        exact = tm[tm["key"] == s["key"]]
        if len(exact) >= 1:
            if has_dob and len(exact) > 1:
                best = exact.iloc[(exact["dob"] - s["dob"]).abs().argsort().values[0]]
            else:
                best = exact.iloc[0]
            if has_dob and pd.notna(best["dob"]) and abs(best["dob"] - s["dob"]) > 365 * DAY:
                warn.append((s["name"], best["name"], str(best["dob"].date()), str(s["dob"].date())))
                fd_age = (today - s["dob"]).days / 365.25
                if 15 <= fd_age <= 45 and abs(best["dob"] - s["dob"]) > 5 * 365 * DAY:
                    best = None  # plausible dob but 5+ years apart: a different person
        elif has_dob:
            cand = tm[(tm["dob"] - s["dob"]).abs() <= DAY]
            if len(cand):
                scores = cand["key"].map(lambda k: name_score(s["key"], k))
                if scores.max() >= 0.6:
                    best = cand.loc[scores.idxmax()]
                    fuzzy.append((s["name"], best["name"], round(scores.max(), 2)))
        elif pd.notna(s["age"]):
            cand = tm[(tm["age"] - s["age"]).abs() <= 0.15]
            if len(cand):
                scores = cand["key"].map(lambda k: name_score(s["key"], k))
                if scores.max() >= 0.85:
                    best = cand.loc[scores.idxmax()]
                    fuzzy.append((s["name"], best["name"], round(scores.max(), 2)))

        if best is None or s["name"] in EXCLUDE:
            unmatched.append(s["name"])
        else:
            out.append({"name": s["name"], "value_m": best["market_value_in_eur"] / 1e6,
                        "tm_id": best["player_id"], "played": s["l_played"] + s["c_played"],
                        **{c: best[c] for c in FEATURE_COLS}})
    return pd.DataFrame(out), fuzzy, warn, unmatched


def build():
    """Create values.csv (target) and tm_features.csv from the raw Kaggle files."""
    if not cfg.STATS_CSV.exists():
        raise SystemExit(f"{cfg.STATS_CSV} not found. Run: python -m plvalue fetch")
    stats = pd.read_csv(cfg.STATS_CSV)
    if "dob" not in stats.columns:
        raise SystemExit("stats.csv has no 'dob' column. Re-run: python -m plvalue fetch")
    stats["key"] = stats["name"].map(norm)
    stats["dob"] = pd.to_datetime(stats["dob"], errors="coerce")

    res, fuzzy, warn, unmatched = match_players(stats, load_transfermarkt())
    res = res.drop_duplicates("name")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res[["name"] + FEATURE_COLS].to_csv(cfg.TM_FEATURES_CSV, index=False)

    # football-data sometimes lists one person under two ids/names (e.g. "Mohamed Cho"
    # and "Mohamed Ali Cho"). Keep only the row with more appearances in the target file.
    dupes = res[res.duplicated("tm_id", keep=False)]["name"].tolist()
    res = res.sort_values("played", ascending=False).drop_duplicates("tm_id")
    res[["name", "value_m"]].to_csv(cfg.VALUES_CSV, index=False)

    print(f"Matched {len(res)} of {len(stats)} players -> {cfg.VALUES_CSV.name} (+ {cfg.TM_FEATURES_CSV.name})")
    print(f"  fuzzy: {len(fuzzy)} | unmatched: {len(unmatched)}")
    if dupes:
        print("Same player listed twice in football-data (kept the one with more appearances):", dupes)
    if fuzzy:
        print("\nFuzzy matches to eyeball (yours -> Transfermarkt, score):")
        for a, b, sc in fuzzy[:50]:
            print(f"  {a} -> {b} ({sc})")
    if warn:
        print("\nSame name but birth dates differ by 1+ year (possibly different people):")
        for a, b, d1, d2 in warn:
            print(f"  {a} -> {b}: Transfermarkt dob {d1}, football-data dob {d2}")
    if unmatched:
        print("\nUnmatched (first 40):", ", ".join(unmatched[:40]))
