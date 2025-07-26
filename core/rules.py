import re
from typing import List, Dict, Any
from collections import Counter

from utils.data_models import TextBlock, EnhancedTextBlock

def apply_rules(blocks: List[Dict[str, Any]]) -> List[EnhancedTextBlock]:
    """
    Applies a set of rules to classify text blocks into structural tags.

    Args:
        blocks: A list of raw text block dictionaries from the parser.

    Returns:
        A list of EnhancedTextBlock objects with assigned tags.
    """
    if not blocks:
        return []

    # --- 1. Font Size Analysis ---
    # Find the most common font size to identify body text
    font_sizes = [block['font_size'] for block in blocks if block['text'].strip()]
    if not font_sizes:
        # Handle case where there's no text
        most_common_font_size = 10.0 
    else:
        most_common_font_size = Counter(font_sizes).most_common(1)[0][0]
    
    heading_font_size_threshold = most_common_font_size * 1.2

    # --- 2. Rule Application ---
    enhanced_blocks = []
    for i, block_data in enumerate(blocks):
        block = TextBlock(**block_data)
        enhanced_block = EnhancedTextBlock(**block.model_dump(), block_id=i)

        text = enhanced_block.text.strip()
        bbox = enhanced_block.bbox

        # Rule 0: Skip empty blocks
        if not text:
            enhanced_block.tag = "empty"
            enhanced_blocks.append(enhanced_block)
            continue

        # Rule 1: Header/Footer Identification (based on vertical position)
        # Assuming a standard A4 page height of ~842 points
        if bbox[1] < 80: # Top ~10% of the page
            enhanced_block.tag = "header"
            enhanced_blocks.append(enhanced_block)
            continue
        if bbox[1] > 700: # Bottom ~10% of the page
            enhanced_block.tag = "footer"
            enhanced_blocks.append(enhanced_block)
            continue

        # Rule 2: List Item Identification (using regex)
        # Matches patterns like '• ...', '1. ...', 'a) ...', '* ...', '- ...', ' ...'
        list_pattern = re.compile(r'^\s*(?:[•*-]|\d+\.|\w\)|)\s+')
        if list_pattern.match(text):
            enhanced_block.tag = "list_item"
            enhanced_blocks.append(enhanced_block)
            continue

        # Rule 3: Noise/Separator Identification (e.g., ToC dots)
        if all(char in ' ._' for char in text):
            enhanced_block.tag = "separator"
            enhanced_blocks.append(enhanced_block)
            continue

        # Rule 4: Heading Identification
        if enhanced_block.font_size > heading_font_size_threshold:
            enhanced_block.tag = "heading"
            enhanced_blocks.append(enhanced_block)
            continue

        # Rule 5: Default to Paragraph
        # If no other rule matched, it's a paragraph.
        enhanced_block.tag = "paragraph"
        enhanced_blocks.append(enhanced_block)

    return enhanced_blocks

