import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Complete 22-feature list for enhanced semantic model
FEATURE_COLUMNS = [
    # Font & Style Features (6)
    'font_size_zscore', 'font_size_percentile', 'font_size_ratio_max', 'font_size_ratio_median',
    'is_bold', 'is_italic',
    
    # Position Features (4)
    'y_position_normalized', 'x_position_normalized', 'space_above_ratio', 'horizontal_alignment',
    
    # Text Content Features (4)
    'text_length_zscore', 'word_count_zscore', 'starts_with_number', 'case_pattern',
    
    # Context Features (3)
    'text_density_around', 'follows_whitespace', 'text_uniqueness',
    
    # Semantic Group Features (5)
    'semantic_type', 'element_count', 'is_bullet_list', 'is_table_content', 'bbox_aspect_ratio'
]

def load_and_validate_dataset(dataset_path="data/enhanced_training_dataset.csv"):
    """
    Load and validate the training dataset with comprehensive checks
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate required columns
        missing_features = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return None
        
        if 'label' not in df.columns:
            logger.error("Missing 'label' column in dataset")
            return None
        
        # Validate labels
        valid_labels = {'Title', 'H1', 'H2', 'H3', 'Body'}
        invalid_labels = set(df['label'].unique()) - valid_labels
        if invalid_labels:
            logger.warning(f"Found invalid labels: {invalid_labels}")
            # Filter out invalid labels
            df = df[df['label'].isin(valid_labels)]
            logger.info(f"Dataset shape after filtering: {df.shape}")
        
        # Handle missing values
        missing_count = df[FEATURE_COLUMNS].isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with zeros")
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
        
        # Handle infinite values
        infinite_count = 0
        for col in FEATURE_COLUMNS:
            if df[col].dtype in ['float64', 'int64']:
                inf_mask = df[col].isin([float('inf'), float('-inf')])
                if inf_mask.any():
                    logger.warning(f"Found infinite values in {col}, replacing with finite values")
                    df.loc[inf_mask, col] = 0
                    infinite_count += inf_mask.sum()
        
        if infinite_count > 0:
            logger.warning(f"Replaced {infinite_count} infinite values")
        
        # Data quality summary
        logger.info(f"Dataset quality check:")
        logger.info(f"  - Rows: {len(df)}")
        logger.info(f"  - Features: {len(FEATURE_COLUMNS)}")
        logger.info(f"  - Missing values: {missing_count}")
        logger.info(f"  - Infinite values: {infinite_count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def prepare_data(df):
    """
    Prepare features and labels for training with enhanced preprocessing
    """
    # Extract features
    X = df[FEATURE_COLUMNS].copy()
    
    # Map labels to integers
    label_mapping = {
        'Body': 0,
        'Title': 1, 
        'H1': 2,
        'H2': 3,
        'H3': 4
    }
    
    y = df['label'].map(label_mapping)
    
    # Validate mapping
    if y.isnull().any():
        logger.error("Some labels could not be mapped to integers")
        return None, None, None
    
    # Feature scaling check
    feature_ranges = {}
    for col in FEATURE_COLUMNS:
        if X[col].dtype in ['float64', 'int64']:
            feature_ranges[col] = {
                'min': X[col].min(),
                'max': X[col].max(),
                'mean': X[col].mean(),
                'std': X[col].std()
            }
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Label distribution:")
    for label, code in label_mapping.items():
        count = (y == code).sum()
        percentage = (count / len(y)) * 100
        logger.info(f"  {label} ({code}): {count} ({percentage:.1f}%)")
    
    # Check for class imbalance
    label_counts = y.value_counts()
    imbalance_ratio = label_counts.max() / label_counts.min() if label_counts.min() > 0 else float('inf')
    if imbalance_ratio > 10:
        logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f})")
    
    return X, y, label_mapping

def create_document_aware_splits(df, y, test_size=0.2, random_state=42):
    """
    Create train/test splits that respect document boundaries to prevent data leakage
    """
    if 'pdf_name' in df.columns:
        # Group by PDF to prevent leakage
        unique_pdfs = df['pdf_name'].unique()
        n_test_pdfs = max(1, int(len(unique_pdfs) * test_size))
        
        np.random.seed(random_state)
        test_pdfs = np.random.choice(unique_pdfs, n_test_pdfs, replace=False)
        
        train_mask = ~df['pdf_name'].isin(test_pdfs)
        test_mask = df['pdf_name'].isin(test_pdfs)
        
        logger.info(f"Document-aware split:")
        logger.info(f"  Training PDFs: {len(unique_pdfs) - n_test_pdfs}")
        logger.info(f"  Testing PDFs: {n_test_pdfs}")
        
        return train_mask, test_mask
    else:
        # Fallback to stratified random split
        logger.warning("No PDF grouping information found, using stratified random split")
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        train_mask = np.isin(indices, train_indices)
        test_mask = np.isin(indices, test_indices)
        
        return train_mask, test_mask

def train_enhanced_lightgbm_model(dataset_path="data/enhanced_training_dataset.csv"):
    """
    Train LightGBM model with enhanced semantic features and document-aware validation
    """
    start_time = time.time()
    logger.info("Starting enhanced LightGBM model training...")
    
    # Load dataset
    df = load_and_validate_dataset(dataset_path)
    if df is None:
        return None
    
    # Prepare data
    X, y, label_mapping = prepare_data(df)
    if X is None:
        return None
    
    # Handle 'semantic_type' column
    if 'semantic_type' in X.columns:
        X = pd.get_dummies(X, columns=['semantic_type'], prefix='semantic_type', dummy_na=False)
    
    # The final list of features after one-hot encoding
    final_feature_columns = X.columns.tolist()

    logger.info(f"Training with {len(final_feature_columns)} features:")
    for i, feature in enumerate(final_feature_columns, 1):
        logger.info(f"{i:2d}. {feature}")
    
    # Create document-aware splits
    train_mask, test_mask = create_document_aware_splits(df, y)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Enhanced model parameters for semantic features
    model_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.08,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 8,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'random_state': 42,
        'verbosity': -1,
        'class_weight': 'balanced'
    }
    
    # Initialize model
    model = lgb.LGBMClassifier(**model_params)
    
    # Train with validation
    logger.info("Training model with validation...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50, show_stdv=False)
        ]
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\nModel Performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    label_names = ['Body', 'Title', 'H1', 'H2', 'H3']
    report = classification_report(y_test, y_pred, target_names=label_names, labels=list(label_mapping.values()), output_dict=True, zero_division=0)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=label_names, labels=list(label_mapping.values()), zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info("Predicted -->")
    logger.info(f"{'Actual':<8} " + " ".join(f"{name:<8}" for name in label_names))
    for i, (true_label, row) in enumerate(zip(label_names, cm)):
        logger.info(f"{true_label:<8} " + " ".join(f"{val:<8}" for val in row))
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': final_feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        logger.info(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # Cross-validation with document awareness
    if 'pdf_name' in df.columns:
        logger.info("\nPerforming document-aware cross-validation...")
        try:
            # Use GroupKFold to respect document boundaries
            gkf = GroupKFold(n_splits=min(5, df['pdf_name'].nunique()))
            cv_scores = cross_val_score(
                model, X, y, cv=gkf, scoring='accuracy',
                groups=df['pdf_name'] if 'pdf_name' in df.columns else None
            )
            logger.info(f"Document-aware CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            logger.warning(f"Document-aware cross-validation failed: {e}")
    
    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    model_path = "models/enhanced_lightgbm_model.txt"
    model.booster_.save_model(model_path)
    
    # Save comprehensive model metadata
    training_time = time.time() - start_time
    metadata = {
        'model_type': 'LightGBM Enhanced',
        'version': '2.0.0',
        'features': final_feature_columns,
        'feature_count': len(final_feature_columns),
        'label_mapping': label_mapping,
        'model_params': model_params,
        'training_date': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(final_feature_columns)
        },
        'performance_metrics': {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': importance_df.to_dict('records')
        },
        'data_quality': {
            'missing_values_handled': True,
            'infinite_values_handled': True,
            'class_imbalance_ratio': float(y.value_counts().max() / y.value_counts().min())
        }
    }
    
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Model size validation
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    logger.info(f"\nModel Training Summary:")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Model file size: {model_size_mb:.1f} MB")
    logger.info(f"Training time: {training_time:.1f} seconds")
    
    # Validate constraints
    constraint_checks = {
        'model_size_ok': model_size_mb <= 200,
        'accuracy_ok': accuracy >= 0.85,
        'training_time_ok': training_time <= 1800  # 30 minutes
    }
    
    logger.info(f"\nConstraint Validation:")
    for check, passed in constraint_checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}: {'PASS' if passed else 'FAIL'}")
    
    if all(constraint_checks.values()):
        logger.info("🎉 All constraints satisfied! Model ready for deployment.")
    else:
        logger.warning("⚠️  Some constraints not met. Review model configuration.")
    
    return model

def validate_model_performance(model_path="models/enhanced_lightgbm_model.txt", 
                              dataset_path="data/enhanced_training_dataset.csv"):
    """
    Validate trained model performance with comprehensive testing
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    logger.info("Validating model performance...")
    
    try:
        # Load model
        model = lgb.Booster(model_file=model_path)
        
        # Load test dataset
        df = load_and_validate_dataset(dataset_path)
        if df is None:
            return False
        
        # Load metadata to get the exact feature set used for training
        metadata_path = "models/model_metadata.json"
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}, cannot validate.")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        model_features = metadata['features']

        X, y, label_mapping = prepare_data(df)
        if X is None:
            return False
        
        # One-hot encode the semantic_type column, same as in training
        if 'semantic_type' in X.columns:
            X = pd.get_dummies(X, columns=['semantic_type'], prefix='semantic_type', dummy_na=False)
        
        # Align columns with the features the model was trained on
        X_aligned = X.reindex(columns=model_features, fill_value=0)

        # Make predictions
        predictions = model.predict(X_aligned.values)
        predicted_classes = predictions.argmax(axis=1)
        prediction_probabilities = predictions.max(axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predicted_classes)
        logger.info(f"Overall model accuracy: {accuracy:.4f}")
        
        # Performance by class
        label_names = ['Body', 'Title', 'H1', 'H2', 'H3']
        report = classification_report(y, predicted_classes, target_names=label_names, labels=list(label_mapping.values()), output_dict=True, zero_division=0)
        
        logger.info("Performance by class:")
        for label in label_names:
            if label in report:
                class_metrics = report[label]
                precision = class_metrics.get('precision', 0)
                recall = class_metrics.get('recall', 0)
                f1 = class_metrics.get('f1-score', 0)
                logger.info(f"  {label:<8}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Confidence analysis
        confidence_stats = {
            'mean_confidence': float(prediction_probabilities.mean()),
            'std_confidence': float(prediction_probabilities.std()),
            'min_confidence': float(prediction_probabilities.min()),
            'max_confidence': float(prediction_probabilities.max())
        }
        
        logger.info(f"Prediction confidence statistics:")
        for key, value in confidence_stats.items():
            logger.info(f"  {key}: {value:.3f}")
        
        # Performance validation
        min_acceptable_accuracy = 0.85
        performance_checks = {
            'accuracy_ok': accuracy >= min_acceptable_accuracy,
            'title_recall_ok': report.get('Title', {}).get('recall', 0) >= 0.8,
            'h1_precision_ok': report.get('H1', {}).get('precision', 0) >= 0.8,
            'overall_f1_ok': report.get('macro avg', {}).get('f1-score', 0) >= 0.8,
            'confidence_ok': confidence_stats['mean_confidence'] >= 0.7
        }
        
        logger.info(f"\nPerformance Validation:")
        for check, passed in performance_checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}: {'PASS' if passed else 'FAIL'}")
        
        validation_passed = all(performance_checks.values())
        
        if validation_passed:
            logger.info("✅ Model validation passed!")
        else:
            logger.warning("❌ Model validation failed!")
        
        return validation_passed
        
    except Exception as e:
        logger.error(f"Error during model validation: {e}")
        return False

def quick_training_pipeline():
    """
    Quick training pipeline for rapid development and testing
    """
    logger.info("Starting quick training pipeline...")
    
    # Check if dataset exists
    dataset_path = "data/enhanced_training_dataset.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run 'python src/train/build_dataset.py' first")
        return None
    
    # Train model
    model = train_enhanced_lightgbm_model(dataset_path)
    
    if model is not None:
        logger.info("Training completed successfully!")
        
        # Validate performance
        if validate_model_performance():
            logger.info("Model validation passed!")
            return model
        else:
            logger.warning("Model validation failed!")
            return None
    else:
        logger.error("Training failed!")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick training mode
        model = quick_training_pipeline()
    else:
        # Full training mode
        model = train_enhanced_lightgbm_model()
        
        if model is not None:
            logger.info("Model training completed successfully!")
            
            # Validate performance
            if validate_model_performance():
                logger.info("Model validation passed!")
            else:
                logger.warning("Model validation failed!")
        else:
            logger.error("Model training failed!")
