import os

import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(feature_df, show_shap=True):
    """
    Train and evaluate multiple ML models on the feature DataFrame.

    Args:
        feature_df (pd.DataFrame): Feature matrix with target column 'winner'
        show_shap (bool): Whether to display the SHAP plot

    Returns:
        best_model: Trained Random Forest classifier
    """
    X = feature_df.drop(columns=["winner"])
    y = feature_df["winner"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        print(classification_report(y_test, y_pred, target_names=["team2 wins", "team1 wins"]))

    best_model = models["Random Forest"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/random_forest.pkl")
    print("Model saved to models/random_forest.pkl")

    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)

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