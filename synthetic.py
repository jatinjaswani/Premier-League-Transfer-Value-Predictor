"""Builds fake stats.csv + Transfermarkt-shaped raw files so the pipeline can be tested offline."""
import numpy as np
import pandas as pd


def write_synthetic(data_dir, n=250, seed=1):
    rng = np.random.default_rng(seed)
    raw, processed = data_dir / "raw", data_dir / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    pos = rng.choice(["Goalkeeper", "Defence", "Midfield", "Offence"], n, p=[.1, .35, .3, .25])
    age = rng.uniform(18, 36, n)
    dob = pd.Timestamp.today().normalize() - pd.to_timedelta((age * 365.25).round(), unit="D")
    names = [f"Player {i}" for i in range(n)]
    team = rng.choice(["Arsenal", "Man City", "Brentford", "Burnley"], n)
    mins = np.where(rng.random(n) < .6, rng.integers(1500, 4200, n), rng.integers(0, 600, n))
    goals = rng.poisson(np.where(pos == "Offence", 6, 1))
    ast = rng.poisson(2, n)

    stats = pd.DataFrame({
        "id": range(n), "name": names, "team": team, "position": pos,
        "age": age.round(1), "dob": dob.strftime("%Y-%m-%d"),
        "l_played": rng.integers(0, 38, n), "l_goals": goals, "l_assists": ast, "l_pens": 0,
        "c_played": 3, "c_goals": 0, "c_assists": 0, "c_pens": 0,
        "wc_played": 0, "wc_goals": 0, "wc_assists": 0, "wc_pens": 0,
        "went_wc": rng.integers(0, 2, n)})
    stats["club_pos"] = stats.team.map({"Arsenal": 1, "Man City": 2, "Brentford": 9, "Burnley": 20})
    stats["club_pts"] = stats.team.map({"Arsenal": 85, "Man City": 80, "Brentford": 55, "Burnley": 30})
    stats["club_gd"] = stats.team.map({"Arsenal": 40, "Man City": 35, "Brentford": 0, "Burnley": -30})
    stats.to_csv(processed / "stats.csv", index=False)

    log_value = (0.5 + 0.00035 * mins + 0.4 * np.log1p(goals + ast)
                 - 0.07 * np.clip(age - 28, 0, None) + 0.015 * stats.club_pts
                 + 0.4 * stats.went_wc + rng.normal(0, 0.3, n))
    value = np.expm1(log_value).clip(0.2)
    ids = range(1000, 1000 + n)

    pd.DataFrame({
        "player_id": ids, "name": names,
        "date_of_birth": dob.strftime("%Y-%m-%d 00:00:00"), "last_season": 2025,
        "contract_expiration_date": (pd.Timestamp("2026-06-30")
                                     + pd.to_timedelta(rng.integers(0, 1800, n), unit="D")
                                     ).strftime("%Y-%m-%d 00:00:00"),
        "international_caps": rng.integers(0, 80, n),
        "international_goals": rng.integers(0, 20, n)}).to_csv(raw / "players.csv", index=False)
    pd.DataFrame({"player_id": ids, "date": "2026-05-01",
                  "market_value_in_eur": (value * 1e6).round()}).to_csv(
        raw / "player_valuations.csv", index=False)

    rows = []
    for i in range(n):
        for j in range(max(1, int(mins[i] // 90))):
            month = 8 + j % 9
            date = f"2025-{month:02d}-{1 + j % 27:02d}" if month <= 12 else f"2026-{month - 12:02d}-{1 + j % 27:02d}"
            rows.append((1000 + i, date, "GB1" if j % 3 else "CL", 0, 0, 0, 0, 90 if mins[i] > 0 else 0))
    pd.DataFrame(rows, columns=["player_id", "date", "competition_id", "yellow_cards", "red_cards",
                                "goals", "assists", "minutes_played"]).to_csv(
        raw / "appearances.csv", index=False)
