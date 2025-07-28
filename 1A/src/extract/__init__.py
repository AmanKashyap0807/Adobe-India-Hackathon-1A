"""
Enhanced PDF Feature Extraction Module

This module provides document-adaptive feature extraction capabilities
for PDF heading detection with semantic block grouping and artifact filtering.
"""
from .features import extract_document_adaptive_features
from .features import (
    extract_document_adaptive_features,
    clean_special_characters,
    is_document_artifact,
    detect_tables,
    merge_spans,
    # calculate_document_statistics  # This function is now inline in extract_document_adaptive_features
)

__version__ = "1.0.0"
__author__ = "PDF Heading Extraction Team"

# Module-level constants
SUPPORTED_FORMATS = ['.pdf']
MAX_FILE_SIZE_MB = 100
DEFAULT_TIMEOUT_SECONDS = 10

# Feature configuration
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

# Label mappings
LABEL_TO_INT = {
    'Body': 0,
    'Title': 1, 
    'H1': 2,
    'H2': 3,
    'H3': 4
}

INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

def get_feature_info():
    """
    Get information about the feature extraction capabilities
    """
    return {
        'total_features': len(FEATURE_COLUMNS),
        'feature_categories': {
            'font_style': 6,
            'position': 4,
            'text_content': 4,
            'context': 3,
            'semantic': 5
        },
        'supported_formats': SUPPORTED_FORMATS,
        'max_file_size_mb': MAX_FILE_SIZE_MB
    }

__all__ = [
    'extract_document_adaptive_features',
    'clean_special_characters', 
    'is_document_artifact',
    'detect_tables',
    'merge_spans',
    'FEATURE_COLUMNS',
    'LABEL_TO_INT',
    'INT_TO_LABEL',
    'get_feature_info'
]
