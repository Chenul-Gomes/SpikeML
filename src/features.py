import os

import pandas as pd


# Unused - kept for reference and potential future use
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
    
    return feature_df

def build_rolling_features(matches_df, stats_df):
    """
    Build features that capture recent performance trends for each team.

    Args:
        matches_df: DataFrame from matches.csv
        stats_df: DataFrame from player_stats.csv

    Returns:
        DataFrame with rolling features for each match
    """
    merged_df = pd.merge(matches_df, stats_df, left_on="url", right_on="match_url", how="left")

    merged_df["actual_team"] = merged_df.apply(
        lambda row: row["team1"] if row["team"] == "team1" else row["team2"],
        axis=1
    )
    merged_df["date"] = pd.to_datetime(merged_df["date"], format="%a, %B %d, %Y")

    numeric_cols = ["rating", "acs", "kills", "deaths", "assists", "kd_diff", "kast", "adr", "hs", "fk", "fd"]
    merged_df["kast"] = pd.to_numeric(merged_df["kast"].str.replace("%", "", regex=False), errors="coerce")
    merged_df["hs"] = pd.to_numeric(merged_df["hs"].str.replace("%", "", regex=False), errors="coerce")

    for col in numeric_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    team_match_stats = merged_df.groupby(["actual_team", "match_url", "date", "score1", "score2", "team1"])[numeric_cols].mean().reset_index()
    team_match_stats["won"] = team_match_stats.apply(
        lambda row: 1 if (
            (row["actual_team"] == row["team1"] and row["score1"] > row["score2"]) or 
            (row["actual_team"] != row["team1"] and row["score2"] > row["score1"])
            ) else 0,
        axis=1)
    team_match_stats["momentum"] = (
        team_match_stats.groupby("actual_team")["won"].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )
    team_match_stats = team_match_stats.sort_values(["actual_team", "date"])

    rolling_stats = (
        team_match_stats.groupby("actual_team")[numeric_cols]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    rolling_stats["actual_team"] = team_match_stats["actual_team"]
    rolling_stats["match_url"] = team_match_stats["match_url"]
    rolling_stats["date"] = team_match_stats["date"]

    rolling_stats = rolling_stats.rename(columns={col: f"rolling_{col}" for col in numeric_cols})

    final_df = pd.merge(
        rolling_stats, matches_df[["url", "team1", "team2", "score1", "score2"]],
        left_on="match_url", right_on="url", how="left"
    )
    final_df["momentum"] = team_match_stats["momentum"].values

    team1_df = final_df[final_df["actual_team"] == final_df["team1"]].copy()
    team2_df = final_df[final_df["actual_team"] == final_df["team2"]].copy()

    rolling_cols = [col for col in final_df.columns if col.startswith("rolling_")]

    team1_df = team1_df.rename(columns={**{col: f"team1_{col}" for col in rolling_cols}, "momentum": "team1_momentum"})
    team2_df = team2_df.rename(columns={**{col: f"team2_{col}" for col in rolling_cols}, "momentum": "team2_momentum"})

    match_features = pd.merge(
        team1_df[["match_url", "team1_momentum"] + [f"team1_{col}" for col in rolling_cols]], 
        team2_df[["match_url", "team2_momentum"] + [f"team2_{col}" for col in rolling_cols]], 
        on="match_url", 
        how="inner"
    )
    
    rolling_feature_df = pd.merge(
        match_features, matches_df[["url", "score1", "score2"]], 
        left_on="match_url", right_on="url", how="left"
    )
    rolling_feature_df["winner"] = (rolling_feature_df["score1"] > rolling_feature_df["score2"]).astype(int)

    drop_cols = ["match_url", "url", "score1", "score2"]
    rolling_feature_df = rolling_feature_df.drop(columns=drop_cols)
    os.makedirs("data/processed", exist_ok=True)
    rolling_feature_df.to_csv("data/processed/rolling_features.csv", index=False)

    return rolling_feature_df
