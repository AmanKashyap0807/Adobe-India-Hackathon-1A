import lightgbm as lgb
import json
import sys
import os
import logging
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.extract.features import extract_document_adaptive_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same feature order as training (22 features)
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

class PDFHeadingExtractor:
    def __init__(self, model_path="models/enhanced_lightgbm_model.txt"):
        """
        Initialize the PDF heading extractor with trained model
        """
        self.model_path = model_path
        self.model = None
        self.label_map = {0: 'Body', 1: 'Title', 2: 'H1', 3: 'H2', 4: 'H3'}
        
        self.load_model()
    
    def load_model(self):
        """
        Load the trained LightGBM model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = lgb.Booster(model_file=self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def extract_features_from_pdf(self, pdf_path):
        """
        Extract document-adaptive features from PDF
        """
        try:
            features_list = extract_document_adaptive_features(pdf_path)
            
            if not features_list:
                logger.warning(f"No features extracted from {pdf_path}")
                return [], []
            
            # Prepare feature matrix
            X = []
            text_blocks = []
            
            for features in features_list:
                # Extract feature values in correct order
                feature_vector = []
                for col in FEATURE_COLUMNS:
                    value = features.get(col, 0)  # Default to 0 if missing
                    feature_vector.append(value)
                
                X.append(feature_vector)
                text_blocks.append({
                    'text': features.get('text', ''),
                    'page_num': features.get('page_num', 0),
                    'semantic_type': features.get('semantic_type', 'text_block'),
                    'element_count': features.get('element_count', 1)
                })
            
            return X, text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting features from {pdf_path}: {e}")
            return [], []
    
    def predict_structure(self, pdf_path):
        """
        Predict PDF structure using trained model
        """
        # Extract features
        X, text_blocks = self.extract_features_from_pdf(pdf_path)
        
        if not X:
            logger.warning(f"No features to predict for {pdf_path}")
            return {'title': '', 'outline': []}
        
        try:
            # Make predictions
            predictions = self.model.predict(X)
            predicted_classes = predictions.argmax(axis=1)
            prediction_probabilities = predictions
            
            # Build output structure
            title = ""
            outline = []
            
            for i, (text_block, pred_class, probs) in enumerate(zip(text_blocks, predicted_classes, prediction_probabilities)):
                label = self.label_map[pred_class]
                confidence = float(probs[pred_class])
                
                if label == 'Title':
                    # Only keep the highest confidence title
                    if not title or confidence > 0.8:  # High confidence threshold for title
                        title = text_block['text']
                elif label in ['H1', 'H2', 'H3']:
                    outline.append({
                        'level': label,
                        'text': text_block['text'],
                        'page': text_block['page_num'] + 1,
                        'confidence': confidence,
                        'semantic_type': text_block.get('semantic_type', 'text_block'),
                        'element_count': text_block.get('element_count', 1)
                    })
            
            # Apply post-processing rules
            outline = self.apply_hierarchy_rules(outline)
            
            # Clean up outline (remove debug info for final output)
            clean_outline = []
            for item in outline:
                clean_outline.append({
                    'level': item['level'],
                    'text': item['text'],
                    'page': item['page']
                })
            
            result = {
                'title': title,
                'outline': clean_outline
            }
            
            logger.info(f"Extracted structure: Title + {len(clean_outline)} headings")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting structure: {e}")
            return {'title': '', 'outline': []}
    
    def apply_hierarchy_rules(self, outline):
        """
        Apply hierarchical consistency rules to improve structure
        """
        if not outline:
            return outline
        
        # Sort by page and position (implicitly by order in list)
        corrected_outline = []
        
        for item in outline:
            current_level = item['level']
            
            # Hierarchy correction logic
            if corrected_outline:
                prev_level = corrected_outline[-1]['level']
                
                # Don't allow H3 to directly follow H1 (should be H2)
                if current_level == 'H3' and prev_level == 'H1':
                    item['level'] = 'H2'
                    logger.info(f"Corrected H3 to H2 after H1: {item['text'][:50]}...")
                
                # Don't allow large level jumps downward
                level_map = {'H1': 1, 'H2': 2, 'H3': 3}
                if (current_level in level_map and prev_level in level_map and
                    level_map[current_level] - level_map[prev_level] > 1):
                    # Reduce to appropriate level
                    new_level_num = level_map[prev_level] + 1
                    if new_level_num <= 3:
                        item['level'] = f'H{new_level_num}'
                        logger.info(f"Corrected {current_level} to H{new_level_num}: {item['text'][:50]}...")
            
            corrected_outline.append(item)
        
        return corrected_outline

def predict_pdf_structure(pdf_path, model_path="models/enhanced_lightgbm_model.txt"):
    """
    Main function for predicting PDF structure
    """
    try:
        extractor = PDFHeadingExtractor(model_path)
        result = extractor.predict_structure(pdf_path)
        return result
    except Exception as e:
        logger.error(f"Failed to predict structure for {pdf_path}: {e}")
        return {'title': '', 'outline': []}

def main():
    """
    Command line interface for PDF structure extraction
    """
    if len(sys.argv) != 2:
        print("Usage: python predict.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Predict structure
    result = predict_pdf_structure(pdf_path)
    
    # Save result
    output_path = Path(pdf_path).with_suffix('.json')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        
        # Print summary
        print(f"\nExtracted Structure:")
        print(f"Title: {result['title']}")
        print(f"Headings: {len(result['outline'])}")
        
        for item in result['outline']:
            print(f"  {item['level']} (p.{item['page']}): {item['text'][:60]}...")
            
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
