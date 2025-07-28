import pandas as pd
import lightgbm as lgb
import fitz  # PyMuPDF
import json
import logging
import os
import sys
from collections import defaultdict

# Add src directory to path to handle relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from extract import extract_document_adaptive_features, FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s')
logger = logging.getLogger(__name__)

def build_outline_from_headings(headings_df):
    """
    Build a hierarchical outline from a DataFrame of detected headings.
    """
    if headings_df.empty:
        return []

    outline = []
    # Sort by page and vertical position if possible
    sort_cols = []
    if 'page_num' in headings_df.columns:
        sort_cols.append('page_num')
    if 'bbox_y0' in headings_df.columns:
        sort_cols.append('bbox_y0')
    
    if sort_cols:
        headings_df = headings_df.sort_values(by=sort_cols)
    
    
    for _, row in headings_df.iterrows():
        # Use text_block if available, otherwise fall back to text
        text_col = 'text_block' if 'text_block' in row else 'text'
        page_col = 'page_num' if 'page_num' in row else 0
        
        outline.append({
            "level": row['role'],
            "text": row[text_col],
            "page": int(row[page_col]) + 1  # 1-based page number for output
        })
    return outline

# Define absolute path for the model inside the container
CONTAINER_MODEL_PATH = "/app/models/enhanced_lightgbm_model.txt"

# Mapping from model output to role names
ROLE_MAPPING = {
    0: 'Body',
    1: 'Title',
    2: 'H1',
    3: 'H2',
    4: 'H3'
}

class PDFHeadingExtractor:
    def __init__(self, model_path=CONTAINER_MODEL_PATH):
        """
        Initialize the PDF heading extractor with trained model
        """
        try:
            self.model = lgb.Booster(model_file=model_path)
            metadata_path = model_path.replace("enhanced_lightgbm_model.txt", "model_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata['features']
            logger.info(f"Model and {len(self.feature_names)} features loaded successfully from {model_path}")
        except (lgb.basic.LightGBMError, FileNotFoundError) as e:
            logger.error(f"Failed to load model or metadata: {e}")
            self.model = None
            self.feature_names = []

    def _extract_features_from_pdf(self, pdf_path):
        """
        Helper to extract features and handle errors
        """
        try:
            # Call the correct function directly with the path
            features, doc_stats = extract_document_adaptive_features(pdf_path)
            if not features:
                return pd.DataFrame(), {}
            
            df = pd.DataFrame(features)
            # One-hot encode the 'semantic_type' column, same as in training
            if 'semantic_type' in df.columns:
                df = pd.get_dummies(df, columns=['semantic_type'], prefix='semantic_type', dummy_na=False)

            return df, doc_stats
        except Exception as e:
            logger.error(f"Error extracting features from {pdf_path}: {e}")
            return pd.DataFrame(), {}

    def predict_structure(self, pdf_path):
        """
        Predict PDF structure using trained model
        """
        if not self.model:
            return {"title": "", "outline": []}

        try:
            features_df, doc_stats = self._extract_features_from_pdf(pdf_path)
            if features_df.empty:
                logger.warning(f"No features extracted from {pdf_path}. Returning empty structure.")
                return {"title": "", "outline": []}

            # Add standard column names that the rest of the code expects
            features_df['text_block'] = features_df['text']  # Rename text to text_block
            
            # Fix column names for bbox coordinates
            if 'bbox' in features_df.columns:
                bbox_values = features_df['bbox'].tolist()
                features_df['bbox_y0'] = [b[1] if isinstance(b, list) and len(b) > 1 else 0 for b in bbox_values]
            else:
                features_df['bbox_y0'] = 0  # Default value if no bbox column
                
            # Align columns with the features the model was trained on
            X_predict = features_df.reindex(columns=self.feature_names, fill_value=0)
            
            # Predict class probabilities and get the most likely class
            predictions = self.model.predict(X_predict)
            predicted_labels = predictions.argmax(axis=1)
            
            features_df['role'] = [ROLE_MAPPING.get(label, 'Body') for label in predicted_labels]

            # Extract title - use available font size column
            title_candidates = features_df[features_df['role'] == 'Title']
            if not title_candidates.empty:
                # Try different possible font size column names
                for col in ['font_size', 'font_size_ratio_max', 'font_size_zscore']:
                    if col in title_candidates.columns:
                        title = title_candidates.nlargest(1, col).iloc[0]['text_block']
                        break
                else:  # No suitable column found
                    title = title_candidates.iloc[0]['text_block']
            else:
                # Fallback: use the highest text block on the first page with largest font
                first_page_blocks = features_df[features_df['page_num'] == 0]  # Changed from 'page' to 'page_num'
                if not first_page_blocks.empty:
                    # Try different possible font size column names
                    for col in ['font_size', 'font_size_ratio_max', 'font_size_zscore']:
                        if col in first_page_blocks.columns:
                            title_block_df = first_page_blocks.nlargest(1, col)
                            if not title_block_df.empty:
                                title_block = title_block_df.iloc[0]
                                # Check if this block is a likely title using a simpler heuristic
                                title = title_block['text_block']
                                break
                    else:  # No suitable column found
                        title = ""
                else:
                    title = ""
            
            # Build the hierarchical outline
            headings = features_df[features_df['role'].isin(['H1', 'H2', 'H3'])]
            outline = build_outline_from_headings(headings)
            
            return {"title": title.strip() if title else "", "outline": outline}

        except Exception as e:
            logger.error(f"Failed to predict structure for {pdf_path}: {e}")
            return {"title": "", "outline": []}


def predict_pdf_structure(pdf_path, model_path=CONTAINER_MODEL_PATH):
    """
    Main function for predicting PDF structure
    """
    extractor = PDFHeadingExtractor(model_path)
    return extractor.predict_structure(pdf_path)