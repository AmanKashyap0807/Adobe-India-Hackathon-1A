import pandas as pd
import glob
import json
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.extract.features import extract_document_adaptive_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_annotation_file(annotation_file):
    """
    Validate that annotation file has the correct structure
    """
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_keys = ['blocks']
        if not all(key in data for key in required_keys):
            logger.warning(f"Missing required keys in {annotation_file}")
            return False
        
        if not isinstance(data['blocks'], list):
            logger.warning(f"'blocks' is not a list in {annotation_file}")
            return False
        
        # Check if blocks have required structure
        for block in data['blocks']:
            if not all(key in block for key in ['id', 'role', 'text']):
                logger.warning(f"Block missing required keys in {annotation_file}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating {annotation_file}: {e}")
        return False

def build_enhanced_training_dataset():
    """
    Build dataset from annotated PDFs with enhanced semantic features
    """
    all_features = []
    all_labels = []
    processing_stats = {
        'total_pdfs_processed': 0,
        'total_blocks_extracted': 0,
        'total_blocks_labeled': 0,
        'artifacts_filtered': 0,
        'semantic_blocks_created': 0
    }
    
    # Get all PDF files in raw directory
    pdf_files = glob.glob("data/raw/*.pdf")
    
    if not pdf_files:
        logger.error("No PDF files found in data/raw/ directory")
        return None
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file)
        annotation_file = f"data/annotated/{Path(pdf_file).stem}.json"
        
        if not os.path.exists(annotation_file):
            logger.warning(f"No annotation file found for {pdf_name}, skipping...")
            continue
        
        if not validate_annotation_file(annotation_file):
            logger.warning(f"Invalid annotation file for {pdf_name}, skipping...")
            continue
        
        try:
            logger.info(f"Processing {pdf_name}...")
            
            # Extract features with enhanced semantic grouping
            features_list = extract_document_adaptive_features(pdf_file)
            
            if not features_list:
                logger.warning(f"No features extracted from {pdf_name}, skipping...")
                continue
            
            processing_stats['total_blocks_extracted'] += len(features_list)
            processing_stats['semantic_blocks_created'] += len(features_list)
            
            # Load annotations
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # Create label mapping: block_id -> role
            label_map = {}
            for block in annotations['blocks']:
                label_map[block['id']] = block['role']
            
            # Match features with labels
            for features in features_list:
                block_id = features.get('block_id', len(all_features))
                
                # Get label, default to 'Body' if not found
                label = label_map.get(block_id, 'Body')
                
                # Validate label
                valid_labels = ['Title', 'H1', 'H2', 'H3', 'Body']
                if label not in valid_labels:
                    logger.warning(f"Invalid label '{label}' found, defaulting to 'Body'")
                    label = 'Body'
                
                all_features.append(features)
                all_labels.append(label)
                processing_stats['total_blocks_labeled'] += 1
            
            processing_stats['total_pdfs_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {pdf_name}: {e}")
            continue
    
    if not all_features:
        logger.error("No features extracted from any PDF files")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    
    # Remove non-feature columns
    feature_columns = [col for col in df.columns if col not in ['text', 'page_num', 'block_id', 'label']]
    
    # Ensure we have all 22 expected features
    expected_features = [
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
    
    # Check for missing features
    missing_features = set(expected_features) - set(feature_columns)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Add missing features with default values
        for feature in missing_features:
            df[feature] = 0
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save dataset
    output_file = "data/enhanced_training_dataset.csv"
    df.to_csv(output_file, index=False)
    
    # Print comprehensive statistics
    logger.info("\n" + "="*50)
    logger.info("DATASET CREATION SUMMARY")
    logger.info("="*50)
    logger.info(f"PDFs processed: {processing_stats['total_pdfs_processed']}")
    logger.info(f"Total blocks extracted: {processing_stats['total_blocks_extracted']}")
    logger.info(f"Total blocks labeled: {processing_stats['total_blocks_labeled']}")
    logger.info(f"Semantic blocks created: {processing_stats['semantic_blocks_created']}")
    
    logger.info(f"\nDataset saved to: {output_file}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features used: {len(expected_features)}")
    
    logger.info("\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    total_labels = len(df)
    
    for label, count in label_counts.items():
        percentage = (count / total_labels) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Check for class imbalance
    min_class_size = label_counts.min()
    max_class_size = label_counts.max()
    imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
    
    if imbalance_ratio > 10:
        logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f})")
        logger.warning("Consider collecting more examples of underrepresented classes")
    
    logger.info("\nSemantic Block Type Distribution:")
    if 'semantic_type' in df.columns:
        semantic_counts = df['semantic_type'].value_counts()
        for stype, count in semantic_counts.items():
            percentage = (count / total_labels) * 100
            logger.info(f"  Type {stype}: {count} ({percentage:.1f}%)")
    
    # Data quality checks
    logger.info("\nData Quality Checks:")
    
    # Check for missing values
    missing_values = df[expected_features].isnull().sum().sum()
    logger.info(f"  Missing values: {missing_values}")
    
    # Check for infinite values
    infinite_values = 0
    for col in expected_features:
        if df[col].dtype in ['float64', 'int64']:
            infinite_values += df[col].isin([float('inf'), float('-inf')]).sum()
    logger.info(f"  Infinite values: {infinite_values}")
    
    if missing_values > 0 or infinite_values > 0:
        logger.warning("Data quality issues detected. Consider data cleaning.")
    
    logger.info("="*50)
    
    return df

def create_balanced_splits(df, test_size=0.2, random_state=42):
    """
    Create balanced train/test splits with stratification
    """
    from sklearn.model_selection import train_test_split
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['text', 'page_num', 'block_id', 'label']]
    X = df[feature_columns]
    y = df['label']
    
    # Create stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"\nTrain/Test Split Created:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Testing samples: {len(X_test)}")
        
        # Check class distribution in splits
        logger.info("\nTraining set distribution:")
        train_dist = y_train.value_counts()
        for label, count in train_dist.items():
            percentage = (count / len(y_train)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        logger.info("\nTesting set distribution:")
        test_dist = y_test.value_counts()
        for label, count in test_dist.items():
            percentage = (count / len(y_test)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return X_train, X_test, y_train, y_test
        
    except ValueError as e:
        logger.error(f"Error creating stratified split: {e}")
        logger.info("Falling back to random split without stratification")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Create enhanced dataset
    dataset = build_enhanced_training_dataset()
    
    if dataset is not None:
        logger.info("Dataset creation completed successfully!")
        
        # Optionally create train/test splits
        try:
            X_train, X_test, y_train, y_test = create_balanced_splits(dataset)
            
            # Save splits
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            train_df.to_csv("data/train_split.csv", index=False)
            test_df.to_csv("data/test_split.csv", index=False)
            
            logger.info("Train/test splits saved successfully!")
            
        except Exception as e:
            logger.error(f"Error creating splits: {e}")
            
    else:
        logger.error("Dataset creation failed!")
