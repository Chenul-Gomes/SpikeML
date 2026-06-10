import os

import pandas as pd

from src.scraper import BASE_URL, scrape_all_player_stats, scrape_match_index
from src.features import build_rolling_features
from src.models import train_model

RESCRAPE = False

if __name__ == "__main__":

    os.makedirs("data/raw", exist_ok=True)

    if RESCRAPE:
        matches_df = pd.DataFrame(scrape_match_index())
        matches_df["url"] = BASE_URL + matches_df["url"]
        matches_df.to_csv("data/raw/matches.csv", index=False)

        stats_df = pd.DataFrame(scrape_all_player_stats(matches_df["url"]))
        stats_df.to_csv("data/raw/player_stats.csv", index=False)
    else:
        matches_df = pd.read_csv("data/raw/matches.csv")
        stats_df = pd.read_csv("data/raw/player_stats.csv")

    feature_df = build_rolling_features(matches_df, stats_df)
    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True)) 

    train_model(feature_df)