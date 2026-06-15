"""
models.py — SpikeML Model Training & Evaluation
=================================================
Trains and evaluates multiple ML models on the feature matrix.
Uses XGBoost for SHAP explainability and Logistic Regression
as the best performing model for predictions.
"""

import os

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(feature_df, show_shap=False, show_metrics=False):
    """
    Train and evaluate multiple ML models on the feature DataFrame.

    Args:
        feature_df (pd.DataFrame): Feature matrix with target column 'winner'
        show_shap (bool): Whether to display the SHAP plot
        show_metrics (bool): Whether to display evaluation metrics

    Returns:
        best_model: Trained Random Forest classifier
    """
    # ── Train/Test Split ──────────────────────────────────────────────────────
    X = feature_df.drop(columns=["winner"])
    y = feature_df["winner"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Model Comparison ──────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: {acc:.2%}")
        if show_metrics:
            print(classification_report(
                y_test, y_pred, target_names=["team2 wins", "team1 wins"]
            ))

    # ── Save Best Model ───────────────────────────────────────────────────────
    # Random Forest performs best with map win rate features
    best_model = models["Logistic Regression"]
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/logistic_regression.pkl")
    joblib.dump(list(X.columns), "models/feature_cols.pkl")
    print("Model saved to models/logistic_regression.pkl")

    # ── SHAP Explainability ───────────────────────────────────────────────────
    # XGBoost used for SHAP as it produces cleaner explanations than Random Forest
    xgb = models["XGBoost"] # reuse already-trained model from comparison loop
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_test)

    plt.close("all")
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False, max_display=24)
    plt.gcf().set_size_inches(14, 12)
    plt.tight_layout()
    plt.savefig("data/processed/shap_importance.png", bbox_inches="tight", dpi=150)
    if show_shap:
        plt.show()
    plt.close("all")

    return best_model

def predict_match(team1, team2):
    """
    Predict win probability for a hypothetical match between two teams.

    Args:
        team1 (str): Name of team 1
        team2 (str): Name of team 2
        matches_df (pd.DataFrame): Full match history
        stats_df (pd.DataFrame): Full player stats history

    Returns:
        float: Probability that team1 wins (0 to 1)
    """
    model = joblib.load("models/logistic_regression.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    final_df = pd.read_csv("data/processed/final_features.csv")  # need to save this in main.py

    # grab most recent row where each team appeared
    team1_matches = final_df[(final_df["team1"] == team1) | (final_df["team2"] == team1)]
    team2_matches = final_df[(final_df["team1"] == team2) | (final_df["team2"] == team2)]

    if team1_matches.empty or team2_matches.empty:
        return None

    team1_row = team1_matches.iloc[-1]
    team2_row = team2_matches.iloc[-1]

    # build input using team1's features as team1 and team2's features as team2
    input_row = {}
    for col in feature_cols:
        if col.startswith("team1_"):
            stat = col.replace("team1_", "")
            input_row[col] = team1_row[f"team1_{stat}"] if f"team1_{stat}" in team1_row.index else 0.5
        elif col.startswith("team2_"):
            stat = col.replace("team2_", "")
            input_row[col] = team2_row[f"team2_{stat}"] if f"team2_{stat}" in team2_row.index else 0.5

    input_df = pd.DataFrame([input_row])[feature_cols]
    input_df = input_df.fillna(0.5)
    prob = model.predict_proba(input_df)[0][1]
    return prob