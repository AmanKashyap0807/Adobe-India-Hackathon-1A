import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import json
from dataset_balancer import DatasetBalancer

# Import the exact feature list from features.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'extract'))
from features import FEATURE_COLUMNS

def train_model():
    """Train LightGBM model on the extracted features with enhanced evaluation (20 features including multi-line)."""
    # Check if dataset exists
    if not os.path.exists("dataset.csv"):
        print("Error: dataset.csv not found. Run build_dataset.py first.")
        return
    
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")
    
    # Check for required features
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features: {missing_features}")
        print("Please rebuild dataset with: python src/train/build_dataset.py")
        return
    
    # Prepare features and labels
    X = df[FEATURE_COLUMNS]
    y = df["label"].map({"Body": 0, "Title": 1, "H1": 2, "H2": 3, "H3": 4})
    
    print(f"Training with {len(FEATURE_COLUMNS)} features (including multi-line features):")
    for i, feature in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i:2d}. {feature}")
    print()
    print(f"Samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Multi-line statistics
    if 'is_multiline' in df.columns:
        multi_line_count = df['is_multiline'].sum()
        multi_line_percentage = (multi_line_count / len(df)) * 100
        avg_line_count = df['line_count'].mean() if 'line_count' in df.columns else 1.0
        print(f"\nMulti-line Statistics:")
        print(f"  Multi-line blocks: {multi_line_count} ({multi_line_percentage:.1f}%)")
        print(f"  Average line count: {avg_line_count:.1f}")
    
    # Check dataset balance
    balancer = DatasetBalancer()
    annotations = [{"role": label} for label in df["label"]]
    print("\n" + balancer.get_balance_report(annotations))
    
    # Split dataset
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=42)
    
    print(f"\nTraining set: {len(X_tr)} samples")
    print(f"Test set: {len(X_te)} samples")
    
    # Train model with enhanced parameters for 20 features
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=7,  # Increased for 20 features
        min_child_samples=10,     # works on small data
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,  # Suppress LightGBM output
        objective='multiclass',
        num_class=5
    )
    
    print("\nTraining LightGBM model with multi-line support...")
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              eval_metric="multi_logloss",
              early_stopping_rounds=30,
              verbose=False)
    
    # Evaluate model
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, 
                               target_names=["Body", "Title", "H1", "H2", "H3"]))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_te, y_pred)
    print("Predicted →")
    print("Actual ↓")
    print("     Body Title   H1   H2   H3")
    for i, label in enumerate(["Body", "Title", "H1", "H2", "H3"]):
        print(f"{label:5} {cm[i]}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Top 15):")
    print(feature_importance.head(15))
    
    # Multi-line feature importance
    multi_line_features = ['line_count', 'is_multiline', 'line_font_consistency']
    multi_line_importance = feature_importance[feature_importance['feature'].isin(multi_line_features)]
    if not multi_line_importance.empty:
        print("\nMulti-line Feature Importance:")
        print(multi_line_importance)
    
    # Save the model
    model.booster_.save_model("model.txt")
    print("\n✓ Model saved as model.txt")
    
    # Save training results
    training_results = {
        "model_info": {
            "algorithm": "LightGBM",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "features_used": len(FEATURE_COLUMNS),
            "multi_line_features": 3
        },
        "dataset_info": {
            "total_samples": len(df),
            "training_samples": len(X_tr),
            "test_samples": len(X_te),
            "features": FEATURE_COLUMNS,
            "label_distribution": df["label"].value_counts().to_dict(),
            "multi_line_statistics": {
                "multi_line_blocks": int(multi_line_count) if 'is_multiline' in df.columns else 0,
                "multi_line_percentage": float(multi_line_percentage) if 'is_multiline' in df.columns else 0,
                "avg_line_count": float(avg_line_count) if 'line_count' in df.columns else 1.0
            }
        },
        "performance": {
            "accuracy": (y_pred == y_te).mean(),
            "feature_importance": feature_importance.to_dict('records'),
            "multi_line_feature_importance": multi_line_importance.to_dict('records') if not multi_line_importance.empty else []
        }
    }
    
    with open("training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)
    
    print("✓ Training results saved to training_results.json")
    
    # Model size check
    model_size = os.path.getsize("model.txt") / 1024  # KB
    print(f"✓ Model size: {model_size:.1f} KB")
    
    if model_size > 500:  # 500 KB limit
        print("⚠️  Warning: Model size exceeds 500 KB limit")
    else:
        print("✓ Model size within limits")
    
    # Feature compatibility verification
    print(f"✓ Feature compatibility: All {len(FEATURE_COLUMNS)} features used")
    print(f"✓ Multi-line support: {len(multi_line_features)} multi-line features included")

if __name__ == "__main__":
    train_model() 