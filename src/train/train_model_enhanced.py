import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import glob
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.features import FEATURE_COLUMNS, extract_document_adaptive_features, is_document_artifact

# Updated feature list to include new semantic features (22 total)
ENHANCED_FEATURE_COLUMNS = [
    # Original 17 features (maintains compatibility)
    'font_size_zscore', 'font_size_percentile', 'font_size_ratio_max', 'font_size_ratio_median',
    'is_bold', 'is_italic', 'y_position_normalized', 'x_position_normalized',
    'space_above_ratio', 'horizontal_alignment', 'text_length_zscore', 'word_count_zscore',
    'starts_with_number', 'case_pattern', 'text_density_around', 'follows_whitespace', 'text_uniqueness',
    
    # Multi-line features (3)
    'line_count', 'is_multiline', 'line_font_consistency',
    
    # NEW: Semantic Group Features (5 features)
    'semantic_type', 'element_count', 'is_bullet_list', 'is_table_content', 'bbox_aspect_ratio'
]

# Add ensure_dir helper function
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

class EnhancedModelTrainer:
    def __init__(self, data_dir="data", output_dir="models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model = None
        self.feature_importance = None
        self.label_map = {
            'Body': 0, 
            'Title': 1, 
            'H1': 2, 
            'H2': 3, 
            'H3': 4
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Create output directory if it doesn't exist
        ensure_dir(output_dir)
        ensure_dir(data_dir)
        ensure_dir(os.path.join(data_dir, "raw"))
        ensure_dir(os.path.join(data_dir, "annotated"))
        
    def build_clean_dataset(self, annotation_dir="data/annotated", raw_dir="data/raw"):
        """Build dataset with proper artifact filtering"""
        print("Building enhanced dataset with artifact filtering...")
        
        # Create directories if they don't exist
        ensure_dir(annotation_dir)
        ensure_dir(raw_dir)
        
        all_features = []
        all_labels = []
        filtering_stats = {
            'total_blocks': 0,
            'artifacts_filtered': 0,
            'content_blocks': 0
        }
        
        annotation_files = glob.glob(os.path.join(annotation_dir, "*.json"))
        if not annotation_files:
            print(f"No annotation files found in {annotation_dir}")
            return None
            
        for annotation_file in annotation_files:
            pdf_name = Path(annotation_file).stem
            pdf_file = os.path.join(raw_dir, f"{pdf_name}")
            
            # Add .pdf extension if not present
            if not pdf_file.lower().endswith('.pdf'):
                pdf_file += '.pdf'
                
            if not os.path.exists(pdf_file):
                print(f"⚠️ PDF file not found: {pdf_file}")
                continue
                
            print(f"Processing {pdf_name}...")
            
            try:
                # Load annotations
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Extract features with artifact filtering
                features_list, doc_stats = extract_document_adaptive_features(pdf_file)
                
                # Track filtering statistics
                filtering_stats['total_blocks'] += doc_stats.get('total_lines', 0)
                filtering_stats['artifacts_filtered'] += doc_stats.get('artifact_lines', 0)
                filtering_stats['content_blocks'] += doc_stats.get('content_lines', 0)
                
                # Get label map (only for content blocks)
                if 'blocks' in annotations:
                    # Legacy format
                    label_map = {block['id']: block['role'] for block in annotations['blocks']}
                elif 'groups' in annotations:
                    # Group-based format
                    label_map = {group['group_id']: group['role'] for group in annotations['groups']}
                else:
                    print(f"⚠️ Unknown annotation format in {annotation_file}")
                    continue
                
                # Match features with labels
                for i, features in enumerate(features_list):
                    label = label_map.get(i, 'Body')  # Default to Body
                    
                    # Skip artifacts if they're explicitly labeled as such
                    if label == 'Artifact':
                        filtering_stats['artifacts_filtered'] += 1
                        continue
                    
                    # Extract only the enhanced feature columns we need (22 features)
                    feature_dict = {col: features.get(col, 0) for col in ENHANCED_FEATURE_COLUMNS}
                    all_features.append(feature_dict)
                    all_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {pdf_name}: {e}")
        
        if not all_features:
            print("No features extracted. Please check your annotation files.")
            return None
        
        # Print filtering statistics
        print(f"\nFiltering Statistics:")
        print(f"Total blocks processed: {filtering_stats['total_blocks']}")
        artifacts_pct = filtering_stats['artifacts_filtered']/filtering_stats['total_blocks']*100 if filtering_stats['total_blocks'] > 0 else 0
        content_pct = filtering_stats['content_blocks']/filtering_stats['total_blocks']*100 if filtering_stats['total_blocks'] > 0 else 0
        print(f"Artifacts filtered: {filtering_stats['artifacts_filtered']} ({artifacts_pct:.1f}%)")
        print(f"Content blocks used: {filtering_stats['content_blocks']} ({content_pct:.1f}%)")
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        df['label'] = all_labels
        
        # Save dataset
        clean_dataset_path = os.path.join(self.data_dir, "clean_training_dataset.csv")
        df.to_csv(clean_dataset_path, index=False)
        
        print(f"Clean dataset created with {len(df)} samples")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        
        # Save dataset statistics
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'filtering_statistics': filtering_stats,
            'feature_columns': [col for col in df.columns if col != 'label'],
        }
        
        with open(os.path.join(self.data_dir, "clean_dataset_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        return df
    
    def train_with_missing_labels(self, df=None):
        """Train model properly handling unlabeled/filtered data"""
        if df is None:
            # Try to load dataset
            try:
                df = pd.read_csv(os.path.join(self.data_dir, "clean_training_dataset.csv"))
            except FileNotFoundError:
                print("Dataset not found. Building dataset first...")
                df = self.build_clean_dataset()
                
        if df is None:
            print("Failed to build or load dataset.")
            return False
        
        print("\n" + "="*50)
        print("TRAINING ENHANCED MODEL WITH ARTIFACT FILTERING")
        print("="*50)
        
        # Ensure all required features are present
        missing_features = [col for col in ENHANCED_FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print("Using available features only.")
        
        available_features = [col for col in ENHANCED_FEATURE_COLUMNS if col in df.columns]
        print(f"Training with {len(available_features)} enhanced features (including semantic grouping):")
        for i, feature in enumerate(available_features, 1):
            print(f"  {i:2d}. {feature}")
        
        # Separate labeled and potentially unlabeled data
        labeled_mask = df['label'].notna() & ~df['label'].isin(['UNLABELED', 'Artifact'])
        
        X_labeled = df[labeled_mask][available_features]
        y_labeled = df[labeled_mask]['label'].map(self.label_map)
        
        # Check for any missing mappings
        if y_labeled.isna().any():
            unknown_labels = df[labeled_mask]['label'][~df[labeled_mask]['label'].isin(self.label_map.keys())].unique()
            print(f"Warning: Unknown labels found: {unknown_labels}")
            print("These will be treated as unlabeled data.")
            labeled_mask = labeled_mask & df['label'].isin(self.label_map.keys())
            X_labeled = df[labeled_mask][available_features]
            y_labeled = df[labeled_mask]['label'].map(self.label_map)
        
        print(f"\nTraining with {len(X_labeled)} labeled samples")
        print("Label distribution:")
        print(y_labeled.map(self.reverse_label_map).value_counts())
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
        )
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train model with class balancing
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            min_child_samples=10,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        cv_scores = cross_val_score(model, X_labeled, y_labeled, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model on all data
        print("\nTraining final model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Classification report
        y_pred = model.predict(X_test)
        class_report = classification_report(
            y_test, 
            y_pred, 
            target_names=[self.reverse_label_map[i] for i in sorted(self.reverse_label_map.keys())]
        )
        
        print("\nClassification Report:")
        print(class_report)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Features by Importance:")
        print(self.feature_importance.head(10))
        
        # Save the model
        model_path = os.path.join(self.output_dir, "enhanced_model.txt")
        model.booster_.save_model(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save feature importance
        self.feature_importance.to_csv(os.path.join(self.output_dir, "feature_importance.csv"), index=False)
        
        # Save evaluation metrics
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_scores": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist()
            },
            "feature_importance": self.feature_importance.to_dict(orient='records'),
            "features_used": available_features,
            "class_balance": y_labeled.map(self.reverse_label_map).value_counts().to_dict(),
            "artifact_filtering": True
        }
        
        with open(os.path.join(self.output_dir, "enhanced_model_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save confusion matrix visualization
        self.plot_confusion_matrix(y_test, y_pred)
        
        self.model = model
        return True
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        labels = [self.reverse_label_map[i] for i in sorted(self.reverse_label_map.keys())]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()
    
    def validate_feature_consistency(self, sample_size=5):
        """Validate that preprocessing doesn't introduce inconsistencies"""
        print("\n" + "="*50)
        print("VALIDATING FEATURE CONSISTENCY")
        print("="*50)
        
        sample_pdfs = glob.glob(os.path.join(self.data_dir, "raw", "*.pdf"))
        if not sample_pdfs:
            print("No PDF files found for validation")
            return
            
        if sample_size > 0 and sample_size < len(sample_pdfs):
            sample_pdfs = sample_pdfs[:sample_size]
        
        results = []
        for pdf_path in sample_pdfs:
            pdf_name = os.path.basename(pdf_path)
            print(f"Validating {pdf_name}...")
            
            try:
                # Method 1: With artifact filtering (clean)
                features_list, doc_stats = extract_document_adaptive_features(pdf_path)
                
                # Extract font sizes from features
                font_sizes_clean = [feature.get('font_size', 0) for feature in features_list]
                
                # Compute clean statistics
                clean_stats = {
                    'median_font': np.median(font_sizes_clean) if font_sizes_clean else 0,
                    'mean_font': np.mean(font_sizes_clean) if font_sizes_clean else 0,
                    'content_blocks': len(features_list),
                    'artifact_blocks': doc_stats.get('artifact_lines', 0),
                    'total_blocks': doc_stats.get('total_lines', 0)
                }
                
                # Compute statistics with artifacts included (dirty)
                dirty_stats = {
                    'median_font': doc_stats.get('median_font_size_with_artifacts', 0),
                    'mean_font': doc_stats.get('mean_font_size_with_artifacts', 0),
                    'total_blocks': doc_stats.get('total_lines', 0)
                }
                
                # Calculate impact
                font_difference = abs(clean_stats['median_font'] - dirty_stats['median_font'])
                artifact_percentage = 0
                if clean_stats['total_blocks'] > 0:
                    artifact_percentage = (clean_stats['artifact_blocks'] / clean_stats['total_blocks']) * 100
                
                results.append({
                    'pdf': pdf_name,
                    'clean_median_font': clean_stats['median_font'],
                    'dirty_median_font': dirty_stats['median_font'],
                    'font_difference': font_difference,
                    'content_blocks': clean_stats['content_blocks'],
                    'artifact_blocks': clean_stats['artifact_blocks'],
                    'total_blocks': clean_stats['total_blocks'],
                    'artifact_percentage': artifact_percentage
                })
                
            except Exception as e:
                print(f"Error validating {pdf_name}: {e}")
        
        if not results:
            print("No validation results generated")
            return
        
        # Create DataFrame with results
        df_results = pd.DataFrame(results)
        
        # Calculate averages
        avg_font_diff = df_results['font_difference'].mean()
        avg_artifact_pct = df_results['artifact_percentage'].mean()
        
        print("\nFeature Consistency Validation Results:")
        print(df_results)
        
        print(f"\nAverage font size difference: {avg_font_diff:.2f}pt")
        print(f"Average artifact percentage: {avg_artifact_pct:.1f}%")
        
        if avg_font_diff > 1.0:
            print("\n⚠️ WARNING: Significant font size differences detected!")
            print("   Artifacts are contaminating document statistics")
            print("   The enhanced model addresses this issue")
        else:
            print("\n✅ Font statistics are consistent")
        
        # Save validation results
        df_results.to_csv(os.path.join(self.output_dir, "feature_consistency.csv"), index=False)

def main():
    trainer = EnhancedModelTrainer()
    
    # Build clean dataset with artifact filtering
    df = trainer.build_clean_dataset()
    
    # Train model with proper handling of labeled/unlabeled data
    trainer.train_with_missing_labels(df)
    
    # Validate feature consistency
    trainer.validate_feature_consistency()

if __name__ == "__main__":
    main()
