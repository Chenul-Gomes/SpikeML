"""
main.py — SpikeML Pipeline Entry Point
=======================================
Orchestrates the full pipeline:
  1. Scrape or load match + player stats data
  2. Build rolling performance features
  3. Build map win rate features
  4. Train and evaluate ML models
"""

import os

import pandas as pd

from src.scraper import BASE_URL, scrape_all_player_stats, scrape_match_index
from src.features import build_rolling_features, build_map_winrates
from src.models import train_model

RESCRAPE = False

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # ── Data Loading ──────────────────────────────────────────────────────────────
    if RESCRAPE:
        matches_df = pd.DataFrame(scrape_match_index())
        matches_df["url"] = BASE_URL + matches_df["url"]
        matches_df.to_csv("data/raw/matches.csv", index=False)

        stats_df = pd.DataFrame(scrape_all_player_stats(matches_df["url"]))
        stats_df.to_csv("data/raw/player_stats.csv", index=False)
    else:
        matches_df = pd.read_csv("data/raw/matches.csv")
        stats_df = pd.read_csv("data/raw/player_stats.csv")

    # ── Feature Engineering ───────────────────────────────────────────────────────
    feature_df = build_rolling_features(matches_df, stats_df)
    map_features = build_map_winrates(matches_df, stats_df)

    # combine rolling stats with map win rates into one feature matrix
    final_df = pd.merge(feature_df, map_features, left_on="match_url", right_on="url", how="left")
    final_df = final_df.drop(columns=["match_url", "url"])
    final_df = final_df.fillna(final_df.mean(numeric_only=True))

    # ── Model Training ────────────────────────────────────────────────────────────
    train_model(final_df)