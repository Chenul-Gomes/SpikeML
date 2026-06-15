"""
app.py — SpikeML Dashboard
===========================
Streamlit dashboard for SpikeML match predictions and historical results.
"""

import pandas as pd
import streamlit as st

from src.models import predict_match

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpikeML", page_icon="🎯", layout="centered")
st.title("🎯 SpikeML — Valorant Match Predictor")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/final_features.csv")

final_df = load_data()
all_teams = sorted(set(final_df["team1"].dropna()) | set(final_df["team2"].dropna()))

# ── Mode Selection ────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Mode", ["Predict a Match", "Browse History"])

# ── Mode 1: Predict ───────────────────────────────────────────────────────────
if mode == "Predict a Match":
    st.subheader("Predict a Match")
    st.write("Select two teams to see their predicted win probabilities.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams, key="team1")
    with col2:
        team2 = st.selectbox("Team 2", all_teams, key="team2")

    if st.button("Predict"):
        if team1 == team2:
            st.warning("Select two different teams.")
        else:
            prob = predict_match(team1, team2)
            if prob is None:
                st.error("Not enough match history for one or both teams.")
            else:
                st.subheader("Win Probability")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(team1, f"{prob:.1%}")
                with col2:
                    st.metric(team2, f"{1 - prob:.1%}")
                st.progress(prob)

# ── Mode 2: Browse History ────────────────────────────────────────────────────
elif mode == "Browse History":
    st.subheader("Match History")
    st.write("Browse past matches with predicted vs actual results.")

    # load raw matches for dates and tournament context
    matches_df = pd.read_csv("data/raw/matches.csv")

    history = final_df[["match_url", "team1", "team2"]].copy()
    history = history.merge(matches_df[["url", "date", "tournament", "score1", "score2"]], 
                            left_on="match_url", right_on="url", how="left")
    history["actual_winner"] = history.apply(
        lambda row: row["team1"] if int(row["score1"]) > int(row["score2"]) else row["team2"], axis=1
    )

    # optional team filter
    filter_team = st.selectbox("Filter by team (optional)", ["All"] + all_teams)
    if filter_team != "All":
        history = history[(history["team1"] == filter_team) | (history["team2"] == filter_team)]

    st.dataframe(
        history[["date", "tournament", "team1", "score1", "score2", "team2", "actual_winner"]],
        use_container_width=True
    )