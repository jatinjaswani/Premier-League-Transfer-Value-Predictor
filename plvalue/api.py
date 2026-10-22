"""football-data.org client: fetches PL + World Cup stats into data/processed/stats.csv."""
import time

import pandas as pd
import requests

from . import config as cfg
from .utils import age_from_dob


def _get(path, **params):
    """GET with rate-limit handling (free tier allows 10 requests/minute)."""
    headers = {"X-Auth-Token": cfg.get_token()}
    for _ in range(5):
        r = requests.get(f"{cfg.API_BASE}{path}", headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("X-RequestCounter-Reset", 60)) + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Still rate limited after 5 retries")


def _require_token():
    if not cfg.get_token():
        raise SystemExit("Set FOOTBALL_DATA_TOKEN (environment variable or .env file) first.")


def probe():
    """Show which data sources the current API plan can reach."""
    _require_token()
    for label, (comp, season, _) in cfg.SOURCES.items():
        for ep in ("scorers", "teams"):
            try:
                d = _get(f"/competitions/{comp}/{ep}", season=season)
                n = len(d.get("scorers", d.get("teams", [])))
                print(f"OK   {label:5} {comp} {season} /{ep}: {n} items")
            except requests.HTTPError as e:
                print(f"FAIL {label:5} {comp} {season} /{ep}: HTTP {e.response.status_code}")


def _scorers(comp, season):
    try:
        return _get(f"/competitions/{comp}/scorers", season=season, limit=100).get("scorers", [])
    except requests.HTTPError as e:
        print(f"! skipped {comp} {season}: HTTP {e.response.status_code}")
        return []


def _player_row(p, team):
    dob = p.get("dateOfBirth")
    return {"id": p["id"], "name": p["name"], "team": team,
            "position": p.get("position") or "Unknown",
            "age": age_from_dob(dob), "dob": str(dob or "")[:10]}


def fetch():
    _require_token()
    players, stats = {}, {k: {} for k in cfg.SOURCES}

    for key, (comp, season, _) in cfg.SOURCES.items():
        for s in _scorers(comp, season):
            p = s["player"]
            stats[key][p["id"]] = {"played": s.get("playedMatches") or 0,
                                   "goals": s.get("goals") or 0,
                                   "assists": s.get("assists") or 0,
                                   "pens": s.get("penalties") or 0}
            if comp == "PL":
                players[p["id"]] = _player_row(p, s.get("team", {}).get("shortName", ""))

    # Squads add defenders/keepers who never appear in scorer tables.
    try:
        for t in _get("/competitions/PL/teams", season=cfg.SOURCES["cur"][1])["teams"]:
            for p in t.get("squad", []):
                players.setdefault(p["id"], _player_row(p, t["shortName"]))
    except requests.HTTPError:
        print("! squads unavailable on this plan: table only has players who scored/assisted")

    # World Cup squads: flag everyone selected, not only scorers.
    wc_ids = set()
    try:
        comp, season, _ = cfg.SOURCES["wc"]
        for t in _get(f"/competitions/{comp}/teams", season=season)["teams"]:
            wc_ids.update(p["id"] for p in t.get("squad", []))
    except requests.HTTPError:
        print("! World Cup squads unavailable: went_wc falls back to scorers only")

    # Last completed season's league table -> club-strength features.
    table = {}
    try:
        std = _get("/competitions/PL/standings", season=cfg.SOURCES["last"][1])["standings"]
        for r in next(x for x in std if x.get("type") == "TOTAL")["table"]:
            table[r["team"]["shortName"]] = (r["position"], r["points"], r["goalDifference"])
    except (requests.HTTPError, StopIteration, KeyError):
        print("! standings unavailable: club-strength features will be constant")

    rows = []
    for pid, info in players.items():
        row = dict(info)
        for key, (_, _, pre) in cfg.SOURCES.items():
            st = stats[key].get(pid, {})
            for col in ("played", "goals", "assists", "pens"):
                row[f"{pre}_{col}"] = st.get(col, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["went_wc"] = (df["id"].isin(wc_ids) | (df["wc_played"] > 0)).astype(int)

    worst = (20, min((v[1] for v in table.values()), default=0),
             min((v[2] for v in table.values()), default=0))
    for i, col in enumerate(("club_pos", "club_pts", "club_gd")):
        df[col] = df["team"].map(lambda t: table.get(t, worst)[i])
    promoted = sorted(set(df.loc[~df["team"].isin(table.keys()), "team"]))
    if table and promoted:
        print(f"  not in last season's table (bottom-of-table values used): {', '.join(promoted)}")

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.STATS_CSV, index=False)
    print(f"Saved {len(df)} PL players -> {cfg.STATS_CSV} ({int(df.went_wc.sum())} in World Cup squads)")
