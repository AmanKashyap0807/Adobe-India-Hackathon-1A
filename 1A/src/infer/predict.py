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

from extract.features import extract_document_adaptive_features, FEATURE_COLUMNS
from .batch_predict import build_outline_from_headings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

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

            # Align columns with the features the model was trained on
            X_predict = features_df.reindex(columns=self.feature_names, fill_value=0)
            
            # Predict class probabilities and get the most likely class
            predictions = self.model.predict(X_predict)
            predicted_labels = predictions.argmax(axis=1)
            
            features_df['role'] = [ROLE_MAPPING.get(label, 'Body') for label in predicted_labels]

            # Extract title
            title_candidates = features_df[features_df['role'] == 'Title']
            if not title_candidates.empty:
                title = title_candidates.loc[title_candidates['font_size'].idxmax()]['text_block']
            else:
                # Fallback: use the highest text block on the first page with largest font
                first_page_blocks = features_df[features_df['page'] == 0]
                if not first_page_blocks.empty:
                    # Heuristic: use nlargest to safely get the row with the largest font size
                    title_block_df = first_page_blocks.nlargest(1, 'font_size')
                    
                    if not title_block_df.empty:
                        title_block = title_block_df.iloc[0]
                        # Check if this block is a likely title
                        if title_block['font_size'] > doc_stats.get('median_font_size', 0) + 2 and \
                           title_block['word_count'] < 20:
                            title = title_block['text_block']
                        else:
                            title = ""
                    else:
                        title = ""
                else:
                    title = ""
            
            # Build the hierarchical outline
            headings = features_df[features_df['role'].isin(['H1', 'H2', 'H3'])]
            outline = build_outline_from_headings(headings)
            
            return {"title": title.strip(), "outline": outline}

        except Exception as e:
            logger.error(f"Failed to predict structure for {pdf_path}: {e}")
            return {"title": "", "outline": []}


def predict_pdf_structure(pdf_path, model_path=CONTAINER_MODEL_PATH):
    """
    Main function for predicting PDF structure
    """
    extractor = PDFHeadingExtractor(model_path)
    return extractor.predict_structure(pdf_path)