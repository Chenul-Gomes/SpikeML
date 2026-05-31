import os

import pandas as pd

from scraper import BASE_URL, scrape_all_player_stats, scrape_match_index

if __name__ == "__main__":

    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(scrape_match_index())
    df["url"] = BASE_URL + df["url"]
    df.to_csv("data/raw/matches.csv", index=False)

    df_stats = pd.DataFrame(scrape_all_player_stats(df["url"]))
    df_stats.to_csv("data/raw/player_stats.csv", index=False)
    print(f"Done {len(df_stats)} rows saved.")

    df_stats = pd.read_csv("data/raw/player_stats.csv")
    print(f"Matches: {len(df)}")
    print(f"Player stat rows: {len(df_stats)}")
    print(f"Unique players: {df_stats['name'].nunique()}")
    print(df_stats.head())