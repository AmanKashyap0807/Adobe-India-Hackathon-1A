import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train_model():
    """Train LightGBM model on the extracted features."""
    # Check if dataset exists
    if not os.path.exists("dataset.csv"):
        print("Error: dataset.csv not found. Run build_dataset.py first.")
        return
    
    df = pd.read_csv("dataset.csv")
    X = df.drop(columns=["label", "text"])
    y = df["label"].map({"Body": 0, "Title": 1, "H1": 2, "H2": 3, "H3": 4})

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        min_child_samples=10,     # works on small data
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              eval_metric="multi_logloss",
              early_stopping_rounds=30,
              verbose=False)

    print("Model Training Results:")
    print(classification_report(y_te, model.predict(X_te)))
    
    # Save the model
    model.booster_.save_model("model.txt")
    print("Model saved as model.txt")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    train_model() 