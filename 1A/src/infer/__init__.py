"""
Production Inference Module

This module provides production-ready inference capabilities for PDF heading
extraction using pre-trained models with document-adaptive features.
"""

from .predict import (
    PDFHeadingExtractor,
    predict_pdf_structure,
    FEATURE_COLUMNS
)

__version__ = "1.0.0"
__author__ = "PDF Heading Extraction Team"

# Inference configuration
INFERENCE_CONFIG = {
    'default_model_path': 'models/enhanced_lightgbm_model.txt',
    'max_processing_time_seconds': 10,
    'max_file_size_mb': 100,
    'confidence_threshold': 0.5,
    'enable_hierarchy_correction': True
}

# Performance targets
PERFORMANCE_TARGETS = {
    'accuracy': 0.88,
    'processing_time_per_page': 0.2,  # seconds
    'memory_usage_mb': 100,
    'model_size_mb': 50
}

# Output format specification
OUTPUT_FORMAT = {
    'title': 'string',
    'outline': [
        {
            'level': 'string (H1|H2|H3)',
            'text': 'string',
            'page': 'integer'
        }
    ]
}

def get_inference_info():
    """
    Get information about inference capabilities and configuration
    """
    return {
        'inference_config': INFERENCE_CONFIG,
        'performance_targets': PERFORMANCE_TARGETS,
        'output_format': OUTPUT_FORMAT,
        'feature_count': len(FEATURE_COLUMNS)
    }

def validate_inference_setup(model_path=None):
    """
    Validate that the inference environment is properly configured
    """
    import os
    import lightgbm as lgb
    
    if model_path is None:
        model_path = INFERENCE_CONFIG['default_model_path']
    
    checks = {
        'model_file_exists': os.path.exists(model_path),
        'lightgbm_available': True,
        'model_loadable': False,
        'model_size_compliant': False
    }
    
    try:
        # Test LightGBM availability
        lgb.Booster(model_file=model_path)
        checks['lightgbm_available'] = True
        checks['model_loadable'] = True
        
        # Check model size
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            checks['model_size_compliant'] = model_size_mb <= 200
            checks['model_size_mb'] = model_size_mb
            
    except Exception as e:
        checks['lightgbm_available'] = False
        checks['error'] = str(e)
    
    return checks

class InferenceError(Exception):
    """Custom exception for inference-related errors"""
    pass

class ModelLoadError(InferenceError):
    """Exception raised when model cannot be loaded"""
    pass

class ProcessingTimeoutError(InferenceError):
    """Exception raised when processing exceeds time limit"""
    pass

__all__ = [
    'PDFHeadingExtractor',
    'predict_pdf_structure',
    'FEATURE_COLUMNS',
    'INFERENCE_CONFIG',
    'PERFORMANCE_TARGETS',
    'OUTPUT_FORMAT',
    'get_inference_info',
    'validate_inference_setup',
    'InferenceError',
    'ModelLoadError',
    'ProcessingTimeoutError'
]
