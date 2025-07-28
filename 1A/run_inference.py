#!/usr/bin/env python3
"""
Main entry point for PDF heading extraction system
Handles batch processing of PDFs in Docker environment
"""

import os
import sys
import json
import logging
from pathlib import Path
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.infer.predict import predict_pdf_structure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_single_pdf(input_path, output_path, model_path="models/enhanced_lightgbm_model.txt"):
    """
    Process a single PDF file and save results
    """
    try:
        logger.info(f"Processing: {input_path}")
        start_time = time.time()
        
        # Extract structure
        result = predict_pdf_structure(input_path, model_path)
        
        # Save result
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed: {output_path} (took {processing_time:.2f}s)")
        
        # Log summary
        title_found = "✓" if result['title'] else "✗"
        heading_count = len(result['outline'])
        logger.info(f"  Title: {title_found}, Headings: {heading_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False

def process_batch_pdfs(input_dir="/app/input", output_dir="/app/output", model_path="models/enhanced_lightgbm_model.txt"):
    """
    Process all PDFs in input directory (Docker environment)
    """
    # Ensure directories exist
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return True
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    success_count = 0
    total_start_time = time.time()
    
    for pdf_file in pdf_files:
        # Generate output filename
        output_filename = pdf_file.stem + ".json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process PDF
        if process_single_pdf(str(pdf_file), output_path, model_path):
            success_count += 1
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info("="*50)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total PDFs: {len(pdf_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(pdf_files) - success_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per PDF: {total_time / len(pdf_files):.2f}s")
    
    # Performance check
    max_time_per_pdf = 10.0  # Hackathon constraint
    avg_time = total_time / len(pdf_files) if pdf_files else 0
    
    if avg_time <= max_time_per_pdf:
        logger.info(f"✓ Performance within constraint (<= {max_time_per_pdf}s per PDF)")
    else:
        logger.warning(f"✗ Performance exceeds constraint ({avg_time:.2f}s > {max_time_per_pdf}s)")
    
    return success_count == len(pdf_files)

def main():
    """
    Main entry point - handles both single file and batch processing
    """
    if len(sys.argv) == 2:
        # Single file mode
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        
        output_path = Path(pdf_path).with_suffix('.json')
        success = process_single_pdf(pdf_path, str(output_path))
        sys.exit(0 if success else 1)
        
    elif len(sys.argv) == 1:
        # Batch mode (Docker environment)
        logger.info("Starting batch PDF processing...")
        
        # Check model exists
        model_path = "models/enhanced_lightgbm_model.txt"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please ensure the model is properly trained and saved")
            sys.exit(1)
        
        # Process all PDFs
        success = process_batch_pdfs()
        sys.exit(0 if success else 1)
        
    else:
        print("Usage:")
        print("  Single file: python run_inference.py <pdf_path>")
        print("  Batch mode:  python run_inference.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
