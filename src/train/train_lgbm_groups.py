import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import json
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.features import FEATURE_COLUMNS

class GroupAwareTrainer:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.group_performance = {}
        
    def train_with_group_awareness(self, dataset_path="data/training_dataset_groups.csv"):
        """Train LightGBM model with group-aware validation and enhanced analysis."""
        
        print("🚀 Training LightGBM Model with Group-Aware Validation")
        print("=" * 60)
        
        # Load dataset
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            print("💡 Please run build_dataset_groups.py first")
            return None
            
        df = pd.read_csv(dataset_path)
        print(f"📊 Loaded dataset with {len(df)} samples")
        
        # Validate features
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return None
        
        print(f"✅ All {len(FEATURE_COLUMNS)} features present")
        
        # Prepare features and labels
        X = df[FEATURE_COLUMNS]
        y = df['label'].map({
            'Body': 0,
            'Title': 1,
            'H1': 2,
            'H2': 3,
            'H3': 4
        })
        
        # Group-aware train-test split
        print("\n📊 Performing Group-Aware Train-Test Split...")
        X_train, X_test, y_train, y_test, train_indices, test_indices = self.group_aware_split(
            X, y, df, test_size=0.2, random_state=42
        )
        
        print(f"📈 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Train model
        print("\n🤖 Training LightGBM Model...")
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,  # Increased for 20 features
            min_child_samples=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='multiclass',
            num_class=5,
            verbose=-1
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        print("\n📊 Model Evaluation...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Model Accuracy: {accuracy:.4f}")
        
        # Generate comprehensive report
        self.generate_training_report(
            df, X_train, X_test, y_train, y_test, y_pred, 
            train_indices, test_indices, accuracy
        )
        
        # Save model
        model_path = "models/lightgbm_model_groups.txt"
        os.makedirs("models", exist_ok=True)
        self.model.booster_.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        return self.model
    
    def group_aware_split(self, X, y, df, test_size=0.2, random_state=42):
        """Perform group-aware train-test split to prevent data leakage."""
        
        # Get unique PDFs
        unique_pdfs = df['pdf_name'].unique()
        np.random.seed(random_state)
        
        # Split PDFs instead of individual samples
        test_pdfs = np.random.choice(
            unique_pdfs, 
            size=int(len(unique_pdfs) * test_size), 
            replace=False
        )
        
        # Create train/test masks
        train_mask = ~df['pdf_name'].isin(test_pdfs)
        test_mask = df['pdf_name'].isin(test_pdfs)
        
        # Split data
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Get indices for detailed analysis
        train_indices = df[train_mask].index.tolist()
        test_indices = df[test_mask].index.tolist()
        
        print(f"📁 Training PDFs: {len(unique_pdfs) - len(test_pdfs)}")
        print(f"📁 Test PDFs: {len(test_pdfs)}")
        
        return X_train, X_test, y_train, y_test, train_indices, test_indices
    
    def generate_training_report(self, df, X_train, X_test, y_train, y_test, y_pred, 
                               train_indices, test_indices, accuracy):
        """Generate comprehensive training report with group analysis."""
        
        # Basic metrics
        label_names = ['Body', 'Title', 'H1', 'H2', 'H3']
        classification_rep = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Multi-line feature analysis
        multi_line_features = ['line_count', 'is_multiline', 'line_font_consistency']
        multi_line_importance = importance_df[importance_df['feature'].isin(multi_line_features)]
        
        # Group performance analysis
        test_df = df.iloc[test_indices].copy()
        test_df['predicted'] = y_pred
        test_df['actual'] = y_test
        
        group_performance = self.analyze_group_performance(test_df)
        
        # Create comprehensive report
        report = {
            "model_info": {
                "model_type": "LightGBM with Group-Aware Validation",
                "features_used": len(FEATURE_COLUMNS),
                "feature_list": FEATURE_COLUMNS,
                "accuracy": accuracy,
                "model_size_kb": os.path.getsize("models/lightgbm_model_groups.txt") / 1024 if os.path.exists("models/lightgbm_model_groups.txt") else 0
            },
            "dataset_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "unique_pdfs": df['pdf_name'].nunique(),
                "training_pdfs": df.iloc[train_indices]['pdf_name'].nunique(),
                "test_pdfs": df.iloc[test_indices]['pdf_name'].nunique()
            },
            "label_distribution": {
                "overall": df['label'].value_counts().to_dict(),
                "training": df.iloc[train_indices]['label'].value_counts().to_dict(),
                "test": df.iloc[test_indices]['label'].value_counts().to_dict()
            },
            "multi_line_statistics": {
                "total_multi_line_groups": len(df[df['is_multiline'] == 1]),
                "multi_line_in_training": len(df.iloc[train_indices][df.iloc[train_indices]['is_multiline'] == 1]),
                "multi_line_in_test": len(df.iloc[test_indices][df.iloc[test_indices]['is_multiline'] == 1]),
                "average_line_count": df['line_count'].mean(),
                "font_consistency_mean": df['line_font_consistency'].mean()
            },
            "feature_importance": {
                "top_15_features": importance_df.head(15).to_dict('records'),
                "multi_line_feature_importance": multi_line_importance.to_dict('records'),
                "feature_categories": {
                    "font_style": [f for f in FEATURE_COLUMNS if any(x in f for x in ['font', 'bold', 'italic'])],
                    "position": [f for f in FEATURE_COLUMNS if any(x in f for x in ['position', 'space', 'alignment'])],
                    "text_content": [f for f in FEATURE_COLUMNS if any(x in f for x in ['text', 'word', 'number', 'case'])],
                    "context": [f for f in FEATURE_COLUMNS if any(x in f for x in ['density', 'whitespace', 'uniqueness'])],
                    "multi_line": [f for f in FEATURE_COLUMNS if any(x in f for x in ['line_count', 'multiline', 'consistency'])]
                }
            },
            "classification_report": classification_rep,
            "group_performance": group_performance,
            "efficiency_analysis": {
                "group_based_training": True,
                "group_aware_split": True,
                "estimated_annotation_efficiency": df['line_count'].mean(),
                "multi_line_accuracy": group_performance.get('multi_line_accuracy', 0)
            }
        }
        
        # Save report
        report_file = "models/training_groups_report.json"
        os.makedirs("models", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n📊 Training Report saved to: {report_file}")
        print(f"🎯 Model Accuracy: {accuracy:.4f}")
        print(f"📈 Multi-line Groups: {report['multi_line_statistics']['total_multi_line_groups']}")
        print(f"📊 Average Line Count: {report['multi_line_statistics']['average_line_count']:.2f}")
        
        # Print top features
        print(f"\n🏆 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Print multi-line feature importance
        if not multi_line_importance.empty:
            print(f"\n🔗 Multi-line Feature Importance:")
            for i, row in multi_line_importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return report
    
    def analyze_group_performance(self, test_df):
        """Analyze performance specifically for multi-line groups."""
        
        # Overall performance by group type
        single_line_acc = accuracy_score(
            test_df[test_df['is_multiline'] == 0]['actual'],
            test_df[test_df['is_multiline'] == 0]['predicted']
        ) if len(test_df[test_df['is_multiline'] == 0]) > 0 else 0
        
        multi_line_acc = accuracy_score(
            test_df[test_df['is_multiline'] == 1]['actual'],
            test_df[test_df['is_multiline'] == 1]['predicted']
        ) if len(test_df[test_df['is_multiline'] == 1]) > 0 else 0
        
        # Performance by line count
        line_count_performance = {}
        for line_count in test_df['line_count'].unique():
            mask = test_df['line_count'] == line_count
            if mask.sum() > 0:
                acc = accuracy_score(test_df[mask]['actual'], test_df[mask]['predicted'])
                line_count_performance[f"{line_count}_lines"] = {
                    "accuracy": acc,
                    "count": mask.sum()
                }
        
        return {
            "single_line_accuracy": single_line_acc,
            "multi_line_accuracy": multi_line_acc,
            "line_count_performance": line_count_performance,
            "multi_line_groups_in_test": len(test_df[test_df['is_multiline'] == 1]),
            "single_line_groups_in_test": len(test_df[test_df['is_multiline'] == 0])
        }

def main():
    """Main function to train the group-aware model."""
    
    print("🚀 Group-Aware LightGBM Training")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = "data/training_dataset_groups.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Please run build_dataset_groups.py first")
        return
    
    # Train model
    trainer = GroupAwareTrainer()
    model = trainer.train_with_group_awareness(dataset_path)
    
    if model is not None:
        print(f"\n✅ Training completed successfully!")
        print(f"💾 Model saved to: models/lightgbm_model_groups.txt")
        print(f"📊 Report saved to: models/training_groups_report.json")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main() 