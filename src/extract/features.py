import fitz  # PyMuPDF
import numpy as np
import re
from collections import Counter
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional

# Define the exact feature list for consistency across all files (now 22 features with semantic grouping)
FEATURE_COLUMNS = [
    # Font & Style Features (6)
    'font_size_zscore',
    'font_size_percentile', 
    'font_size_ratio_max',
    'font_size_ratio_median',
    'is_bold',
    'is_italic',
    
    # Position Features (4)
    'y_position_normalized',
    'x_position_normalized',
    'space_above_ratio',
    'horizontal_alignment',
    
    # Text Content Features (4)
    'text_length_zscore',
    'word_count_zscore',
    'starts_with_number',
    'case_pattern',
    
    # Context Features (3)
    'text_density_around',
    'follows_whitespace',
    'text_uniqueness',
    
    # Multi-line specific features (3)
    'line_count',
    'is_multiline',
    'line_font_consistency',
    
    # NEW: Semantic Group Features (5 features)
    'semantic_type',
    'element_count',
    'is_bullet_list',
    'is_table_content',
    'bbox_aspect_ratio'
]

# Update the is_document_artifact function signature to work with both styles of calls
def is_document_artifact(line_info, page_stats=None, block_position=None):
    """
    Identify document artifacts (headers, footers, page numbers) based on rules.
    Returns True if the line is likely an artifact.
    
    Compatible with both:
    - Direct call with line_info containing all needed data
    - Legacy call with separate page_stats and block_position
    """
    # Handle the case where line_info is passed directly
    if isinstance(line_info, dict) and 'text' in line_info:
        text = line_info['text'].strip()
        bbox = line_info.get('bbox', [0, 0, 0, 0])
        page_height = line_info.get('page_height', 842)  # Default A4
    # Handle legacy format with separate parameters
    else:
        text = page_stats.get('text', '').strip() if page_stats else ''
        bbox = block_position if block_position else [0, 0, 0, 0]
        page_height = page_stats.get('page_height', 842) if page_stats else 842
    
    # Rule 1: Page numbers or short, numeric text
    if re.match(r'^\s*p(age|\.)?\s*\d+\s*$', text, re.IGNORECASE):
        return True
    if re.match(r'^\s*\d+\s*$', text) and len(text) < 5:
        return True
    if re.match(r'^\s*\d+\s+of\s+\d+\s*$', text, re.IGNORECASE):
        return True
    
    # Rule 2: Position-based header/footer detection (top 8% or bottom 8% of page)
    y_position_norm = bbox[1] / page_height if page_height > 0 else 0
    if y_position_norm < 0.08 or y_position_norm > 0.92:
        # Shorter text in headers/footers is common
        if len(text.split()) < 7:
            return True
    
    # Rule 3: Common header/footer patterns
    if re.match(r'^\s*(confidential|draft|copyright|all\s+rights\s+reserved)', text, re.IGNORECASE):
        return True
    
    return False

def clean_special_characters(text: str) -> str:
    """
    Handle special characters and symbols without breaking parsing
    """
    if not text:
        return ""
    
    # Handle common PDF encoding issues
    replacements = {
        '\uf0b7': '•',  # Bullet point from symbol fonts
        '\uf0a7': '•',  # Another bullet variant
        '\uf020': ' ',  # Space from symbol fonts
        '\uf061': 'a',  # Common character mapping issues
        '\uf065': 'e',
        '\uf06f': 'o',
        '\uf074': 't',
        '\uf0e0': 'à',  # Accented characters
        '\uf0e8': 'è',
        '\uf0ec': 'ì',
        '\uf0f2': 'ò',
        '\uf0f9': 'ù',
    }
    
    clean_text = text
    for old, new in replacements.items():
        clean_text = clean_text.replace(old, new)
    
    # Handle Wingdings and similar fonts by converting to Unicode
    try:
        # Normalize Unicode characters
        clean_text = unicodedata.normalize('NFKD', clean_text)
        
        # Remove non-printable characters but keep useful symbols
        clean_text = ''.join(char for char in clean_text 
                           if unicodedata.category(char)[0] != 'C' or char in ['\n', '\t'])
        
    except Exception as e:
        logging.warning(f"Character cleaning issue: {e}")
    
    return clean_text

def extract_vector_bullets(page) -> List[Dict]:
    """
    Extract bullet points that are stored as vector graphics
    """
    bullets = []
    try:
        drawings = page.get_drawings()
        
        for drawing in drawings:
            rect = drawing.get('rect')
            if rect and rect.width < 10 and rect.height < 10:  # Small circular/square shapes
                # Check if it's likely a bullet point
                if rect.width == rect.height or abs(rect.width - rect.height) < 2:
                    bullets.append({
                        'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                        'type': 'vector_bullet',
                        'page_num': page.number
                    })
    except Exception as e:
        logging.warning(f"Error extracting vector bullets: {e}")
    
    return bullets

def detect_and_group_tables(pdf_path: str) -> List[Dict]:
    """
    Detect and group table cells as unified blocks
    """
    tables = []
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Extract all text elements on this page
        page_elements = []
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        clean_text = clean_special_characters(span["text"])
                        if clean_text.strip():
                            page_elements.append({
                                'text': clean_text.strip(),
                                'bbox': span['bbox'],
                                'x0': span['bbox'][0],
                                'y0': span['bbox'][1],
                                'x1': span['bbox'][2],
                                'y1': span['bbox'][3],
                                'font_size': span['size'],
                                'page_num': page_num
                            })
        
        # Group elements by y-coordinate (table rows)
        y_groups = group_by_y_coordinate(page_elements)
        
        # Detect table-like patterns
        for row_group in y_groups:
            if len(row_group) >= 3 and is_table_like_alignment(row_group):
                # Combine into single table block
                table_text = " | ".join([elem['text'] for elem in row_group])
                min_x = min(elem['x0'] for elem in row_group)
                min_y = min(elem['y0'] for elem in row_group)
                max_x = max(elem['x1'] for elem in row_group)
                max_y = max(elem['y1'] for elem in row_group)
                
                tables.append({
                    'page_num': page_num,
                    'text': table_text,
                    'bbox': [min_x, min_y, max_x, max_y],
                    'type': 'table_detected',
                    'cell_count': len(row_group),
                    'row_count': 1,
                    'font_size': np.mean([elem['font_size'] for elem in row_group]),
                    'is_bold': any('Bold' in str(elem.get('font_name', '')) for elem in row_group),
                    'is_italic': any('Italic' in str(elem.get('font_name', '')) for elem in row_group),
                    'page_height': page_height,
                    'page_width': page_width
                })
    
    doc.close()
    return tables

def group_by_y_coordinate(elements: List[Dict], tolerance: float = 5) -> List[List[Dict]]:
    """
    Group text elements by similar y-coordinates (table rows)
    """
    if not elements:
        return []
    
    # Sort by y-coordinate
    sorted_elements = sorted(elements, key=lambda x: x['y0'])
    
    groups = []
    current_group = [sorted_elements[0]]
    
    for elem in sorted_elements[1:]:
        if abs(elem['y0'] - current_group[-1]['y0']) <= tolerance:
            current_group.append(elem)
        else:
            groups.append(current_group)
            current_group = [elem]
    
    groups.append(current_group)
    return groups

def is_table_like_alignment(row_group: List[Dict]) -> bool:
    """
    Check if a group of elements looks like a table row
    """
    if len(row_group) < 3:
        return False
    
    # Check for consistent spacing
    x_positions = sorted([elem['x0'] for elem in row_group])
    spacings = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
    
    if len(spacings) > 1:
        spacing_cv = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 1
        return spacing_cv < 0.5  # Consistent spacing
    
    return False

def extract_document_adaptive_features(pdf_path):
    """
    Enhanced feature extraction with semantic block grouping for:
    - Bullet points and their content
    - Table cells as unified blocks  
    - Special character handling
    - Maintaining document-adaptive statistics
    """
    
    doc = fitz.open(pdf_path)
    all_raw_elements = []
    
    # Step 1: Extract all raw text elements with enhanced character handling
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Extract vector graphics (for bullet points)
        vector_bullets = extract_vector_bullets(page)
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Enhanced character handling for special symbols
                        clean_text = clean_special_characters(span["text"])
                        
                        if clean_text.strip():
                            element = {
                                'page_num': page_num,
                                'text': clean_text,
                                'bbox': span['bbox'],
                                'font_size': span['size'],
                                'font_name': span['font'],
                                'is_bold': "Bold" in span['font'] or "Black" in span['font'],
                                'is_italic': "Italic" in span['font'] or "Oblique" in span['font'],
                                'page_height': page_height,
                                'page_width': page_width,
                                'original_span': span  # Keep for debugging
                            }
                            all_raw_elements.append(element)
    
    doc.close()
    
    # Step 2: Detect and extract tables as unified blocks
    table_blocks = detect_and_group_tables(pdf_path)
    
    # Step 3: Group related elements semantically BEFORE calculating statistics
    semantic_groups = group_elements_semantically(all_raw_elements, vector_bullets, table_blocks)
    
    # Step 4: CRITICAL - Calculate document statistics from individual elements (not groups)
    # This preserves your document-adaptive approach
    doc_stats = calculate_document_statistics(all_raw_elements)
    
    # Step 5: Extract features from semantic groups using individual element statistics
    features_list = []
    for i, group in enumerate(semantic_groups):
        features = extract_group_features(group, doc_stats, semantic_groups, i)
        features['block_id'] = i
        features_list.append(features)
    
    return features_list, doc_stats

def extract_group_features(semantic_group: Dict, doc_stats: Dict, all_groups: List[Dict], 
                          group_index: int) -> Dict:
    """
    Extract features from semantic groups using document statistics from individual elements
    """
    # Your existing 17 features + enhanced features for semantic groups
    text = semantic_group['text']
    font_size = semantic_group['font_size']
    bbox = semantic_group['bbox']
    page_height = semantic_group['page_height']
    page_width = semantic_group['page_width']
    
    # Original document-adaptive features (use doc_stats from individual elements)
    features = {
        # Font & Style Features (6 features) - UNCHANGED
        'font_size_zscore': (font_size - doc_stats['mean_font_size']) / doc_stats['std_font_size'],
        'font_size_percentile': np.searchsorted(np.sort([doc_stats['mean_font_size']] * 100), font_size) / 100,
        'font_size_ratio_max': font_size / doc_stats['max_font_size'],
        'font_size_ratio_median': font_size / doc_stats['median_font_size'],
        'is_bold': 1 if semantic_group['is_bold'] else 0,
        'is_italic': 1 if semantic_group['is_italic'] else 0,
        
        # Position Features (4 features) - UNCHANGED  
        'y_position_normalized': bbox[1] / page_height,
        'x_position_normalized': bbox[0] / page_width,
        'space_above_ratio': calculate_space_above_ratio(semantic_group, all_groups, group_index, doc_stats),
        'horizontal_alignment': calculate_alignment(bbox, page_width),
        
        # Text Content Features (4 features) - UNCHANGED
        'text_length_zscore': (len(text) - doc_stats['mean_text_length']) / doc_stats['std_text_length'],
        'word_count_zscore': (len(text.split()) - doc_stats['mean_word_count']) / doc_stats['std_word_count'],
        'starts_with_number': 1 if bool(re.match(r'^\s*\d+(\.\d+)*\.?\s+', text)) else 0,
        'case_pattern': get_case_pattern(text),
        
        # Context Features (3 features) - ENHANCED
        'text_density_around': calculate_text_density_around(semantic_group, all_groups),
        'follows_whitespace': calculate_follows_whitespace(semantic_group, all_groups, group_index),
        'text_uniqueness': calculate_text_uniqueness(text, all_groups),
        
        # Multi-line Features (3 features)
        'line_count': semantic_group.get('element_count', 1),
        'is_multiline': 1 if semantic_group.get('element_count', 1) > 1 else 0,
        'line_font_consistency': 1.0,  # Default for semantic groups
        
        # NEW: Semantic Group Features (5 features)
        'semantic_type': encode_semantic_type(semantic_group.get('semantic_type', 'text_block')),
        'element_count': semantic_group.get('element_count', 1),
        'is_bullet_list': 1 if semantic_group.get('semantic_type') == 'bullet_list' else 0,
        'is_table_content': 1 if semantic_group.get('semantic_type') in ['table', 'table_detected'] else 0,
        'bbox_aspect_ratio': (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1),
        
        # Keep for annotation/debugging
        'text': text,
        'page_num': semantic_group['page_num']
    }
    
    return features

def calculate_space_above_ratio(current_group, all_groups, current_index, doc_stats):
    """Calculate space above current group relative to document statistics"""
    if current_index == 0:
        return 0
    
    prev_group = all_groups[current_index - 1]
    if prev_group['page_num'] != current_group['page_num']:
        return 0
    
    space_above = current_group['bbox'][1] - prev_group['bbox'][3]
    median_font = doc_stats['median_font_size']
    return max(0, space_above) / median_font

def calculate_alignment(bbox, page_width):
    """Calculate horizontal alignment: 0=left, 1=center, 2=right"""
    center_x = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    
    if abs(center_x - page_center) < page_width * 0.1:
        return 1  # Center
    elif center_x > page_center:
        return 2  # Right
    else:
        return 0  # Left

def get_case_pattern(text):
    """Get case pattern: 0=lowercase, 1=mixed, 2=uppercase, 3=title"""
    if not text:
        return 0
    
    words = text.split()
    if not words:
        return 0
    
    upper_count = sum(1 for word in words if word.isupper())
    lower_count = sum(1 for word in words if word.islower())
    title_count = sum(1 for word in words if word.istitle())
    
    if upper_count == len(words):
        return 2  # All uppercase
    elif lower_count == len(words):
        return 0  # All lowercase
    elif title_count == len(words):
        return 3  # Title case
    else:
        return 1  # Mixed case

def calculate_text_density_around(semantic_group, all_groups):
    """Calculate text density around the current group"""
    nearby_blocks = 0
    bbox = semantic_group['bbox']
    page_num = semantic_group['page_num']
    
    for other_group in all_groups:
        if other_group['page_num'] == page_num:
            distance = abs(other_group['bbox'][1] - bbox[1])
            if distance < 100 and other_group != semantic_group:
                nearby_blocks += 1
    
    return min(nearby_blocks / 5.0, 1.0)  # Normalize to 0-1

def calculate_follows_whitespace(semantic_group, all_groups, group_index):
    """Calculate if the group follows whitespace"""
    if group_index == 0:
        return 0
    
    prev_group = all_groups[group_index - 1]
    if prev_group['page_num'] != semantic_group['page_num']:
        return 0
    
    space_above = semantic_group['bbox'][1] - prev_group['bbox'][3]
    avg_font_size = semantic_group['font_size']
    return 1 if space_above > avg_font_size * 1.5 else 0

def calculate_text_uniqueness(text, all_groups):
    """Calculate how unique this text is within the document"""
    text_lower = text.lower().strip()
    same_text_count = sum(1 for group in all_groups 
                         if group['text'].lower().strip() == text_lower)
    
    return 1.0 - (same_text_count - 1) / len(all_groups)

def group_elements_semantically(all_elements: List[Dict], vector_bullets: List[Dict], 
                               table_blocks: List[Dict]) -> List[Dict]:
    """
    Group text elements semantically while preserving individual element statistics
    """
    if not all_elements:
        return []
    
    # Sort elements in reading order
    sorted_elements = sort_reading_order(all_elements)
    
    semantic_groups = []
    i = 0
    
    while i < len(sorted_elements):
        current_elem = sorted_elements[i]
        
        # Check for bullet point groups
        bullet_group, consumed = detect_bullet_group(sorted_elements, i, vector_bullets)
        if bullet_group:
            semantic_block = create_semantic_block(bullet_group, 'bullet_list')
            if semantic_block:
                semantic_groups.append(semantic_block)
            i += consumed
            continue
        
        # Check for multi-line groups
        multiline_group, consumed = detect_multiline_group(sorted_elements, i)
        if consumed > 1:
            semantic_block = create_semantic_block(multiline_group, 'text_block')
            if semantic_block:
                semantic_groups.append(semantic_block)
            i += consumed
            continue
        
        # Single element
        semantic_block = create_semantic_block([current_elem], 'text_block')
        if semantic_block:
            semantic_groups.append(semantic_block)
        i += 1
    
    # Add table blocks
    for table in table_blocks:
        semantic_groups.append(table)
    
    # Sort final groups in reading order
    return sort_reading_order(semantic_groups)

def detect_bullet_group(elements: List[Dict], start_idx: int, vector_bullets: List[Dict]) -> Tuple[Optional[List[Dict]], int]:
    """
    Detect bullet point and its content
    """
    if start_idx >= len(elements):
        return None, 0
    
    current_elem = elements[start_idx]
    
    # Check if current element is a bullet point
    is_bullet = (
        # Text-based bullets
        re.match(r'^\s*[•▪▫‣⁃◦‧⁌⁍➤➢➣]\s*', current_elem['text']) or
        re.match(r'^\s*[-*+]\s+', current_elem['text']) or
        re.match(r'^\s*\d+[\.\)]\s+', current_elem['text']) or
        re.match(r'^\s*[a-zA-Z][\.\)]\s+', current_elem['text']) or
        # Vector bullets nearby
        has_vector_bullet_nearby(current_elem, vector_bullets)
    )
    
    if not is_bullet:
        return None, 0
    
    # Collect bullet point and its content
    group = [current_elem]
    consumed = 1
    
    # Look for continuation text
    for j in range(start_idx + 1, min(start_idx + 5, len(elements))):  # Look ahead max 5 elements
        next_elem = elements[j]
        
        # Check if this continues the bullet content
        if should_merge_with_bullet(current_elem, next_elem, elements[start_idx:j+1]):
            group.append(next_elem)
            consumed += 1
        else:
            break
    
    return group, consumed

def has_vector_bullet_nearby(text_elem: Dict, vector_bullets: List[Dict], proximity: float = 20) -> bool:
    """
    Check if there's a vector bullet near this text element
    """
    elem_bbox = text_elem['bbox']
    
    for bullet in vector_bullets:
        bullet_bbox = bullet['bbox']
        
        # Check if bullet is to the left and vertically aligned
        if (bullet_bbox[2] < elem_bbox[0] and  # Bullet is to the left
            abs(bullet_bbox[1] - elem_bbox[1]) < proximity):  # Vertically aligned
            return True
    
    return False

def should_merge_with_bullet(bullet_elem: Dict, candidate_elem: Dict, 
                           previous_elements: List[Dict]) -> bool:
    """
    Determine if an element should be merged with a bullet point
    """
    # Same page
    if bullet_elem['page_num'] != candidate_elem['page_num']:
        return False
    
    # Not too far vertically (within 2 line heights)
    vertical_gap = candidate_elem['bbox'][1] - bullet_elem['bbox'][3]
    if vertical_gap > bullet_elem['font_size'] * 2:  # More than 2 line heights apart
        return False
    
    # Must have similar or slightly more indentation than bullet
    bullet_indent = bullet_elem['bbox'][0]
    candidate_indent = candidate_elem['bbox'][0]
    
    # Content should be indented more than or equal to bullet
    if candidate_indent < bullet_indent - bullet_elem['font_size'] * 1.3:
        return False
    
    return True

def detect_multiline_group(elements: List[Dict], start_idx: int) -> Tuple[List[Dict], int]:
    """
    Detect multi-line text blocks (headings, paragraphs)
    """
    if start_idx >= len(elements):
        return [elements[start_idx]], 1
    
    group = [elements[start_idx]]
    consumed = 1
    
    # Look for continuation lines
    for j in range(start_idx + 1, min(start_idx + 4, len(elements))):
        current_elem = elements[start_idx]
        candidate_elem = elements[j]
        
        if should_merge_multiline(current_elem, candidate_elem):
            group.append(candidate_elem)
            consumed += 1
        else:
            break
    
    return group, consumed

def should_merge_multiline(base_elem: Dict, candidate_elem: Dict) -> bool:
    """
    Determine if elements should be merged into multi-line block
    """
    # Same page
    if base_elem['page_num'] != candidate_elem['page_num']:
        return False
    
    # Similar font properties
    font_size_ratio = candidate_elem['font_size'] / base_elem['font_size']
    if not (0.8 <= font_size_ratio <= 1.2):
        return False
    
    # Close vertical spacing (less than 1.5x line height)
    vertical_gap = candidate_elem['bbox'][1] - base_elem['bbox'][3]
    avg_font_size = (base_elem['font_size'] + candidate_elem['font_size']) / 2
    if vertical_gap > avg_font_size * 1.5:
        return False
    
    # Similar horizontal alignment (for headings) or reasonable indentation (for paragraphs)
    horizontal_diff = abs(candidate_elem['bbox'][0] - base_elem['bbox'][0])
    if horizontal_diff > 20:  # More than 20pt difference in alignment
        return False
    
    # Text continuation indicators
    base_text = base_elem['text'].strip()
    if base_text and not base_text.endswith(('.', '!', '?', ':')):
        return True  # Previous line doesn't end with sentence punctuation
    
    return False

def create_semantic_block(element_group: List[Dict], block_type: str) -> Optional[Dict]:
    """
    Create a unified semantic block from grouped elements
    """
    if not element_group:
        return None
    
    # Combine text from all elements
    combined_text = ' '.join(elem['text'].strip() for elem in element_group if elem['text'].strip())
    
    # Calculate bounding box encompassing all elements
    min_x = min(elem['bbox'][0] for elem in element_group)
    min_y = min(elem['bbox'][1] for elem in element_group)
    max_x = max(elem['bbox'][2] for elem in element_group)
    max_y = max(elem['bbox'][3] for elem in element_group)
    
    # Use properties from the first (primary) element
    primary_elem = element_group[0]
    
    return {
        'page_num': primary_elem['page_num'],
        'text': combined_text,
        'bbox': [min_x, min_y, max_x, max_y],
        'font_size': primary_elem['font_size'],
        'font_name': primary_elem['font_name'],
        'is_bold': primary_elem['is_bold'],
        'is_italic': primary_elem['is_italic'],
        'page_height': primary_elem['page_height'],
        'page_width': primary_elem['page_width'],
        'semantic_type': block_type,
        'element_count': len(element_group),
        'original_elements': element_group  # Keep for debugging
    }

def sort_reading_order(elements: List[Dict]) -> List[Dict]:
    """
    Sort elements in reading order (top to bottom, left to right)
    """
    return sorted(elements, key=lambda x: (x['page_num'], x['bbox'][1], x['bbox'][0]))

def calculate_document_statistics(all_elements: List[Dict]) -> Dict:
    """
    CRITICAL: Calculate statistics from individual elements, not semantic groups
    This preserves the document-adaptive approach
    """
    if not all_elements:
        return {}
    
    font_sizes = [elem['font_size'] for elem in all_elements]
    text_lengths = [len(elem['text']) for elem in all_elements]
    word_counts = [len(elem['text'].split()) for elem in all_elements]
    y_positions = [elem['bbox'][1] for elem in all_elements]
    
    return {
        'median_font_size': np.median(font_sizes),
        'mean_font_size': np.mean(font_sizes),
        'std_font_size': np.std(font_sizes) if len(font_sizes) > 1 else 1,
        'max_font_size': max(font_sizes),
        'min_font_size': min(font_sizes),
        'mean_text_length': np.mean(text_lengths),
        'std_text_length': np.std(text_lengths) if len(text_lengths) > 1 else 1,
        'mean_word_count': np.mean(word_counts),
        'std_word_count': np.std(word_counts) if len(word_counts) > 1 else 1,
        'font_size_percentiles': np.percentile(font_sizes, [25, 50, 75, 90, 95])
    }

def calculate_document_statistics_from_lines(all_lines):
    """Calculate document-wide statistics from ALL individual lines (Stage 1)"""
    
    font_sizes = [line['font_size'] for line in all_lines]
    text_lengths = [len(line['text']) for line in all_lines]
    word_counts = [len(line['text'].split()) for line in all_lines]
    y_positions = [line['bbox'][1] for line in all_lines]
    
    # Calculate spacing between consecutive lines
    spacings = []
    for i in range(1, len(all_lines)):
        if all_lines[i]['page_num'] == all_lines[i-1]['page_num']:
            spacing = all_lines[i]['bbox'][1] - all_lines[i-1]['bbox'][3]
            spacings.append(max(0, spacing))
    
    stats = {
        # Font statistics (from ALL individual lines)
        'median_font_size': np.median(font_sizes),
        'mean_font_size': np.mean(font_sizes),
        'std_font_size': np.std(font_sizes) if len(font_sizes) > 1 else 1,
        'max_font_size': max(font_sizes),
        'min_font_size': min(font_sizes),
        
        # Text length statistics (from ALL individual lines)
        'mean_text_length': np.mean(text_lengths),
        'std_text_length': np.std(text_lengths) if len(text_lengths) > 1 else 1,
        'mean_word_count': np.mean(word_counts),
        'std_word_count': np.std(word_counts) if len(word_counts) > 1 else 1,
        
        # Position statistics (from ALL individual lines)
        'mean_y_position': np.mean(y_positions),
        'std_y_position': np.std(y_positions) if len(y_positions) > 1 else 1,
        
        # Spacing statistics (from ALL individual lines)
        'median_spacing': np.median(spacings) if spacings else 10,
        'mean_spacing': np.mean(spacings) if spacings else 10,
        
        # Font hierarchy (from ALL individual lines)
        'font_size_percentiles': np.percentile(font_sizes, [25, 50, 75, 90, 95])
    }
    
    return stats

def group_text_lines_semantically(all_lines):
    """Group lines semantically while preserving original statistics (Stage 2)"""
    
    if not all_lines:
        return []
    
    grouped_blocks = []
    current_group = [all_lines[0]]
    
    for i in range(1, len(all_lines)):
        current_line = all_lines[i]
        previous_line = all_lines[i-1]
        
        # Calculate vertical distance between lines
        vertical_gap = current_line['bbox'][1] - previous_line['bbox'][3]
        
        # Calculate average line height for context
        prev_line_height = previous_line['bbox'][3] - previous_line['bbox'][1]
        curr_line_height = current_line['bbox'][3] - current_line['bbox'][1]
        avg_line_height = (prev_line_height + curr_line_height) / 2
        
        # Grouping criteria
        should_group = (
            # Lines are close together (less than 1.5x line height)
            vertical_gap < avg_line_height * 1.5 and
            # Same font properties (size, family, style)
            current_line['font_size'] == previous_line['font_size'] and
            current_line['font_name'] == previous_line['font_name'] and
            current_line['is_bold'] == previous_line['is_bold'] and
            current_line['is_italic'] == previous_line['is_italic'] and
            # Similar horizontal alignment (within 20px)
            abs(current_line['bbox'][0] - previous_line['bbox'][0]) < 20
        )
        
        if should_group:
            current_group.append(current_line)
        else:
            # Finalize current group and start new one
            if current_group:
                grouped_blocks.append(merge_line_group(current_group))
            current_group = [current_line]
    
    # Don't forget the last group
    if current_group:
        grouped_blocks.append(merge_line_group(current_group))
    
    return grouped_blocks

def merge_line_group(line_group):
    """Merge a group of lines into a single text block."""
    
    if not line_group:
        return None
    
    # Merge text content
    text_parts = [line['text'] for line in line_group]
    merged_text = ' '.join(text_parts).strip()
    
    # Calculate combined bounding box
    x0 = min(line['bbox'][0] for line in line_group)
    y0 = min(line['bbox'][1] for line in line_group)
    x1 = max(line['bbox'][2] for line in line_group)
    y1 = max(line['bbox'][3] for line in line_group)
    merged_bbox = [x0, y0, x1, y1]
    
    # Use properties from first line (they should be similar)
    first_line = line_group[0]
    
    # Calculate font consistency within the group
    font_sizes = [line['font_size'] for line in line_group]
    font_consistency = 1.0 - (np.std(font_sizes) / np.mean(font_sizes)) if np.mean(font_sizes) > 0 else 1.0
    
    return {
        'page_num': first_line['page_num'],
        'text': merged_text,
        'bbox': merged_bbox,
        'font_size': first_line['font_size'],
        'font_name': first_line['font_name'],
        'is_bold': first_line['is_bold'],
        'is_italic': first_line['is_italic'],
        'page_height': first_line['page_height'],
        'page_width': first_line['page_width'],
        'line_count': len(line_group),
        'is_multiline': len(line_group) > 1,
        'line_font_consistency': font_consistency,
        'original_lines': line_group  # Keep reference to original lines for compatibility
    }

def extract_block_features_with_grouping(block, doc_stats, all_blocks, block_index):
    """Extract all 22 features for a grouped block using original document statistics."""
    
    text = block['text']
    font_size = block['font_size']
    bbox = block['bbox']
    page_height = block['page_height']
    page_width = block['page_width']
    line_count = block.get('line_count', 1)
    is_multiline = block.get('is_multiline', 0)
    line_font_consistency = block.get('line_font_consistency', 1.0)
    
    # Get semantic information
    semantic_type = block.get('semantic_type', 'text_block')
    element_count = block.get('element_count', 1)
    
    # ===============================
    # CATEGORY 1: FONT & STYLE FEATURES (6 features) - Using original statistics
    # ===============================
    
    # Feature 1: Font size normalized (Z-score) - from original line statistics
    font_size_zscore = (font_size - doc_stats['mean_font_size']) / doc_stats['std_font_size']
    
    # Feature 2: Font size percentile rank - from original line statistics
    font_size_percentile = np.searchsorted(np.sort([b['font_size'] for b in all_blocks]), font_size) / len(all_blocks)
    
    # Feature 3: Font size ratio to maximum - from original line statistics
    font_size_ratio_max = font_size / doc_stats['max_font_size']
    
    # Feature 4: Font size ratio to median - from original line statistics
    font_size_ratio_median = font_size / doc_stats['median_font_size']
    
    # Feature 5: Is bold
    is_bold = 1 if block['is_bold'] else 0
    
    # Feature 6: Is italic
    is_italic = 1 if block['is_italic'] else 0
    
    # ===============================
    # CATEGORY 2: POSITION FEATURES (4 features) - Using original statistics
    # ===============================
    
    # Feature 7: Vertical position normalized (0 = top, 1 = bottom)
    y_position_normalized = bbox[1] / page_height
    
    # Feature 8: Horizontal position normalized (0 = left, 1 = right)
    x_position_normalized = bbox[0] / page_width
    
    # Feature 9: Space above ratio (compared to median spacing) - from original statistics
    space_above = 0
    if block_index > 0 and all_blocks[block_index-1]['page_num'] == block['page_num']:
        space_above = max(0, bbox[1] - all_blocks[block_index-1]['bbox'][3])
    space_above_ratio = space_above / doc_stats['median_spacing']
    
    # Feature 10: Horizontal alignment (0=left, 1=center, 2=right)
    left_margin = bbox[0]
    center_x = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    
    if abs(center_x - page_center) < page_width * 0.1:  # Within 10% of center
        horizontal_alignment = 1  # Center
    elif left_margin < page_width * 0.2:  # Left margin < 20% of page width
        horizontal_alignment = 0  # Left
    else:
        horizontal_alignment = 2  # Right
    
    # ===============================
    # CATEGORY 3: TEXT CONTENT FEATURES (4 features) - Using original statistics
    # ===============================
    
    # Feature 11: Text length Z-score - from original line statistics
    text_length = len(text)
    text_length_zscore = (text_length - doc_stats['mean_text_length']) / doc_stats['std_text_length']
    
    # Feature 12: Word count Z-score - from original line statistics
    word_count = len(text.split())
    word_count_zscore = (word_count - doc_stats['mean_word_count']) / doc_stats['std_word_count']
    
    # Feature 13: Starts with number
    starts_with_number = 1 if re.match(r'^\d+([.]\d+)*', text) else 0
    
    # Feature 14: Case pattern (0=normal, 1=all_caps, 2=title_case, 3=mixed)
    if text.isupper():
        case_pattern = 1  # All caps
    elif text.istitle():
        case_pattern = 2  # Title case
    elif text.islower():
        case_pattern = 0  # Normal/lowercase
    else:
        case_pattern = 3  # Mixed case
    
    # ===============================
    # CATEGORY 4: CONTEXT FEATURES (3 features)
    # ===============================
    
    # Feature 15: Text density around (blocks within 100px)
    nearby_blocks = 0
    for other_block in all_blocks:
        if other_block['page_num'] == block['page_num']:
            distance = abs(other_block['bbox'][1] - bbox[1])
            if distance < 100 and other_block != block:
                nearby_blocks += 1
    text_density_around = min(nearby_blocks / 5.0, 1.0)  # Normalize to 0-1
    
    # Feature 16: Follows whitespace (large gap before this block)
    follows_whitespace = 1 if space_above_ratio > 1.5 else 0
    
    # Feature 17: Text uniqueness in document
    text_lower = text.lower().strip()
    same_text_count = sum(1 for b in all_blocks if b['text'].lower().strip() == text_lower)
    text_uniqueness = 1.0 - (same_text_count - 1) / len(all_blocks)  # -1 to exclude self
    
    # ===============================
    # CATEGORY 5: MULTI-LINE FEATURES (3 features)
    # ===============================
    
    # Feature 18: Line count (number of lines in this block)
    line_count = line_count
    
    # Feature 19: Is multi-line (0/1 flag)
    is_multiline = is_multiline
    
    # Feature 20: Line font consistency (how consistent fonts are within the group)
    line_font_consistency = line_font_consistency
    
    # ===============================
    # CATEGORY 6: SEMANTIC GROUP FEATURES (5 NEW features)
    # ===============================
    
    # Feature 21: Semantic type (encoded)
    semantic_type_encoded = encode_semantic_type(semantic_type)
    
    # Feature 22: Element count (number of elements grouped together)
    element_count = element_count
    
    # Feature 23: Is bullet list (binary flag)
    is_bullet_list = 1 if semantic_type == 'bullet_list' else 0
    
    # Feature 24: Is table content (binary flag)
    is_table_content = 1 if semantic_type in ['table', 'table_detected'] else 0
    
    # Feature 25: Bounding box aspect ratio (width/height)
    bbox_aspect_ratio = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)
    
    # Return feature dictionary
    return {
        # Font & Style Features
        'font_size_zscore': font_size_zscore,
        'font_size_percentile': font_size_percentile,
        'font_size_ratio_max': font_size_ratio_max,
        'font_size_ratio_median': font_size_ratio_median,
        'is_bold': is_bold,
        'is_italic': is_italic,
        
        # Position Features
        'y_position_normalized': y_position_normalized,
        'x_position_normalized': x_position_normalized,
        'space_above_ratio': space_above_ratio,
        'horizontal_alignment': horizontal_alignment,
        
        # Text Content Features
        'text_length_zscore': text_length_zscore,
        'word_count_zscore': word_count_zscore,
        'starts_with_number': starts_with_number,
        'case_pattern': case_pattern,
        
        # Context Features
        'text_density_around': text_density_around,
        'follows_whitespace': follows_whitespace,
        'text_uniqueness': text_uniqueness,
        
        # Multi-line Features
        'line_count': line_count,
        'is_multiline': is_multiline,
        'line_font_consistency': line_font_consistency,
        
        # NEW: Semantic Group Features
        'semantic_type': semantic_type_encoded,
        'element_count': element_count,
        'is_bullet_list': is_bullet_list,
        'is_table_content': is_table_content,
        'bbox_aspect_ratio': bbox_aspect_ratio,
        
        # Keep text for annotation/debugging
        'text': text,
        'page_num': block['page_num']
    }

# Legacy functions for backward compatibility
def pdf_to_blocks(pdf_path):
    """Extract text blocks from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:       # text only
                yield page_num, block
    doc.close()

def compute_doc_stats(blocks):
    """Compute font size statistics for the document."""
    sizes = [sp["size"] for _, b in blocks for l in b["lines"] for sp in l["spans"]]
    if not sizes:
        return {"median_font": 0, "mean_font": 0, "std_font": 0, "max_font": 0}
    return {
        "median_font": np.median(sizes),
        "mean_font": np.mean(sizes),
        "std_font": np.std(sizes) if len(sizes) > 1 else 0,
        "max_font": max(sizes)
    }

def block_features(block, stats, page_num, page_h=842, page_w=595):
    """Legacy function for backward compatibility"""
    # This is kept for compatibility but should not be used for new code
    # Use extract_document_adaptive_features instead
    features_list, _ = extract_document_adaptive_features("dummy_path")
    if features_list:
        return features_list[0]
    return {}

def encode_semantic_type(semantic_type: str) -> int:
    """Encode semantic type as numerical feature"""
    type_map = {
        'text_block': 0,
        'bullet_list': 1,
        'table': 2,
        'table_detected': 3
    }
    return type_map.get(semantic_type, 0)