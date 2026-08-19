"""Generate 3 domain-diverse sample datasets to prove the agent is generic."""

import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "sample_datasets")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


def sales():
    n = 800
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Furniture", "Clothing", "Groceries"]
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": rng.choice(dates, n),
        "region": rng.choice(regions, n),
        "category": rng.choice(categories, n),
        "units": rng.integers(1, 20, n),
        "unit_price": rng.uniform(5, 500, n).round(2),
    })
    df["revenue"] = (df["units"] * df["unit_price"]).round(2)
    # inject a couple of outliers
    df.loc[rng.integers(0, n, 5), "revenue"] *= 12
    df.sort_values("order_date").to_csv(os.path.join(OUT, "sales.csv"), index=False)


def sports():
    teams = [f"Team_{c}" for c in "ABCDEFGH"]
    rows = []
    for season in [2021, 2022, 2023]:
        for t in teams:
            games = 38
            wins = rng.integers(5, 30)
            draws = rng.integers(0, games - wins)
            rows.append({
                "season": season, "team": t, "games": games,
                "wins": int(wins), "draws": int(draws),
                "losses": int(games - wins - draws),
                "goals_for": int(rng.integers(20, 90)),
                "goals_against": int(rng.integers(20, 90)),
                "points": int(wins * 3 + draws),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sports.csv"), index=False)


def education():
    n = 500
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "gender": rng.choice(["M", "F"], n),
        "grade_level": rng.integers(9, 13, n),
        "study_hours": rng.uniform(0, 40, n).round(1),
        "attendance_pct": rng.uniform(50, 100, n).round(1),
    })
    # exam score correlated with study hours + attendance
    df["exam_score"] = (
        30 + 1.1 * df["study_hours"] + 0.35 * df["attendance_pct"]
        + rng.normal(0, 6, n)
    ).clip(0, 100).round(1)
    df.to_csv(os.path.join(OUT, "education.csv"), index=False)


if __name__ == "__main__":
    sales()
    sports()
    education()
    print("Wrote sample datasets to", OUT)
