from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def train_model(feature_df):
    X = feature_df.drop(columns=["winner"])
    y = feature_df["winner"]   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100,random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=["team2 wins", "team1 wins"]))

    return model