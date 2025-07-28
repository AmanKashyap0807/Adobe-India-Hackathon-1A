"""
Enhanced ML Training Module

This module provides training capabilities for document-adaptive PDF heading
detection using LightGBM with comprehensive feature engineering.
"""

from .build_dataset import (
    build_enhanced_training_dataset,
    validate_annotation_file,
    create_balanced_splits
)

from .train_model import (
    train_enhanced_lightgbm_model,
    load_and_validate_dataset,
    validate_model_performance,
    FEATURE_COLUMNS
)

__version__ = "1.0.0"
__author__ = "PDF Heading Extraction Team"

# Training configuration
TRAINING_CONFIG = {
    'model_type': 'LightGBM',
    'target_accuracy': 0.88,
    'max_training_time_minutes': 30,
    'min_samples_per_class': 50,
    'test_size': 0.2,
    'random_state': 42
}

# Model parameters
LIGHTGBM_PARAMS = {
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
    'verbosity': -1
}

# Dataset requirements
DATASET_REQUIREMENTS = {
    'min_pdfs': 50,
    'optimal_pdfs': 75,
    'min_total_blocks': 3000,
    'optimal_total_blocks': 5000,
    'target_distribution': {
        'Body': 0.70,
        'H2': 0.12,
        'H1': 0.08,
        'H3': 0.07,
        'Title': 0.03
    }
}

def get_training_info():
    """
    Get information about training requirements and configuration
    """
    return {
        'training_config': TRAINING_CONFIG,
        'model_params': LIGHTGBM_PARAMS,
        'dataset_requirements': DATASET_REQUIREMENTS,
        'feature_count': len(FEATURE_COLUMNS)
    }

def validate_training_setup():
    """
    Validate that the training environment is properly configured
    """
    import os
    import lightgbm as lgb
    
    checks = {
        'data_directory': os.path.exists('data/raw'),
        'annotation_directory': os.path.exists('data/annotated'),
        'lightgbm_available': True,
        'models_directory': True
    }
    
    try:
        # Test LightGBM
        lgb.LGBMClassifier(n_estimators=1)
    except Exception:
        checks['lightgbm_available'] = False
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        checks['models_directory'] = True
    
    return checks

__all__ = [
    'build_enhanced_training_dataset',
    'validate_annotation_file',
    'create_balanced_splits',
    'train_enhanced_lightgbm_model',
    'load_and_validate_dataset',
    'validate_model_performance',
    'FEATURE_COLUMNS',
    'TRAINING_CONFIG',
    'LIGHTGBM_PARAMS',
    'DATASET_REQUIREMENTS',
    'get_training_info',
    'validate_training_setup'
]
