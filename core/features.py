from typing import List
import pandas as pd
from collections import Counter

from utils.data_models import EnhancedTextBlock

def create_features(enhanced_blocks: List[EnhancedTextBlock]) -> pd.DataFrame:
    """
    Engineers features for each text block to be used in a machine learning model.

    Args:
        enhanced_blocks: A list of EnhancedTextBlock objects.

    Returns:
        A pandas DataFrame where each row represents a block and each
        column represents a feature.
    """
    if not enhanced_blocks:
        return pd.DataFrame()

    # --- Document-level features ---
    # Calculate the most common font size for normalization
    font_sizes = [block.font_size for block in enhanced_blocks if block.text.strip()]
    if not font_sizes:
        most_common_font_size = 10.0
    else:
        most_common_font_size = Counter(font_sizes).most_common(1)[0][0]

    # Assume A4 page height for relative positioning
    PAGE_HEIGHT = 842 

    # --- Block-level feature extraction ---
    features_list = []
    for block in enhanced_blocks:
        text = block.text.strip()
        
        features = {
            "block_id": block.block_id,
            "rule_based_tag": block.tag,
            
            # Positional Features
            "x_position": block.bbox[0],
            "y_position": block.bbox[1],
            "rel_y_position": block.bbox[1] / PAGE_HEIGHT, # Relative y-position
            
            # Dimensional Features
            "block_width": block.bbox[2] - block.bbox[0],
            "block_height": block.bbox[3] - block.bbox[1],
            
            # Font Features
            "font_size": block.font_size,
            "rel_font_size": block.font_size / most_common_font_size, # Relative font size
            "is_bold": 1 if "bold" in block.font_name.lower() else 0,
            
            # Text-based Features
            "text_length": len(text),
            "line_count": text.count('\n') + 1 if text else 0,
            "is_all_caps": 1 if text.isupper() and len(text) > 1 else 0,
            "ends_with_punct": 1 if text and text[-1] in ".!?" else 0,
            
            # Raw text for potential future use (e.g., text embeddings)
            "text": block.text 
        }
        features_list.append(features)

    return pd.DataFrame(features_list)