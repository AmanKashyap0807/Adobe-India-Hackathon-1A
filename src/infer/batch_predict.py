#!/usr/bin/env python3
"""
Batch processing script for multiple PDFs.
"""

import os
import sys
import json
import glob
from pathlib import Path
from predict import classify

def batch_process(input_dir, output_dir, model_path="model.txt"):
    """Process all PDFs in a directory."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    results = {}
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {os.path.basename(pdf_file)}")
            
            # Classify the PDF
            result = classify(pdf_file, model_path)
            
            # Save individual result
            pdf_name = Path(pdf_file).stem
            output_file = os.path.join(output_dir, f"{pdf_name}.json")
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            results[pdf_name] = {
                "title": result["title"],
                "headings_count": len(result["outline"]),
                "output_file": output_file
            }
            
            print(f"  ✓ Title: {result['title']}")
            print(f"  ✓ Headings: {len(result['outline'])}")
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file}: {e}")
            results[os.path.basename(pdf_file)] = {"error": str(e)}
    
    # Save summary
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBatch processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # Print summary
    successful = sum(1 for r in results.values() if "error" not in r)
    print(f"Successfully processed: {successful}/{len(pdf_files)} files")

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_predict.py <input_directory> <output_directory> [model_path]")
        print("Example: python batch_predict.py ./input_pdfs ./output_results")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else "model.txt"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print("Please train the model first using: python src/train/train_lgbm.py")
        sys.exit(1)
    
    batch_process(input_dir, output_dir, model_path)

if __name__ == "__main__":
    main() 