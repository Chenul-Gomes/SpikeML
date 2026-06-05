import os

import pandas as pd

def build_match_features(matches_df, stats_df):
    """
    Build one row per match with aggregated team features.
    
    Args:
        matches_df: DataFrame from matches.csv
        stats_df: DataFrame from player_stats.csv
    
    Returns:
        DataFrame with one row per match and team features
    """
    merged_df = pd.merge(matches_df, stats_df, left_on="url", right_on="match_url", how="left")

    merged_df["kast"] = pd.to_numeric(merged_df["kast"].str.replace("%", "", regex=False), errors="coerce")
    merged_df["hs"] = pd.to_numeric(merged_df["hs"].str.replace("%", "", regex=False), errors="coerce")
    merged_df["kd_diff"] = pd.to_numeric(merged_df["kd_diff"], errors="coerce")

    numeric_cols = ["rating", "acs", "kills", "deaths", "assists", "kd_diff", "kast", "adr", "hs", "fk", "fd"]
    for col in numeric_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
        
    grouped = merged_df.groupby(["match_url", "team"])[numeric_cols].mean()

    pivoted = grouped.unstack(level="team")
    pivoted.columns = [f"{team}_{stat}" for stat, team in pivoted.columns]
    pivoted = pivoted.reset_index()

    merged_score = pd.merge(pivoted, matches_df[["url", "score1", "score2"]], left_on="match_url", right_on="url", how="left")
    merged_score["winner"] = (merged_score["score1"] > merged_score["score2"]).astype(int)

    drop_cols = ["match_url", "url", "score1", "score2"]
    feature_df = merged_score.drop(columns=drop_cols)
    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv("data/processed/features.csv", index=False)

    print(feature_df.shape)
    print(feature_df.head())
    
    return feature_df    