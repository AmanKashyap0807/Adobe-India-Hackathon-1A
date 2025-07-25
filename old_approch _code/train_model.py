import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
DATASET_PATH = "training_data.csv"
MODEL_OUTPUT_PATH = "model.pkl"


def train():
    """Trains the heading classification model."""
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        print("Training data is empty. Please run create_dataset.py first.")
        return

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(objective='multiclass', random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Performance on Test Set ---")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel successfully trained and saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()