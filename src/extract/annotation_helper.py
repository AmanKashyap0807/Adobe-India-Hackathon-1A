import json
import sys
import os
from features import pdf_to_blocks

def extract_blocks_for_annotation(pdf_path):
    """Extract text blocks from PDF and save them for annotation."""
    blocks = list(pdf_to_blocks(pdf_path))
    
    # Create annotation template
    annotation_data = {
        "pdf_file": os.path.basename(pdf_path),
        "blocks": []
    }
    
    for i, (page_num, block) in enumerate(blocks):
        text = " ".join(sp["text"] for l in block["lines"] for sp in l["spans"]).strip()
        if text:  # Only include non-empty blocks
            annotation_data["blocks"].append({
                "id": i,
                "text": text,
                "page": page_num + 1,
                "role": "Body"  # Default role, to be changed during annotation
            })
    
    # Save annotation template
    pdf_name = os.path.basename(pdf_path)
    output_file = f"data/annotated/{pdf_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"Annotation template created: {output_file}")
    print(f"Found {len(annotation_data['blocks'])} text blocks")
    print("\nTo annotate:")
    print("1. Open the JSON file in any text editor")
    print("2. Change 'role' values to: Title, H1, H2, H3, or Body")
    print("3. Save the file")

def main():
    if len(sys.argv) != 2:
        print("Usage: python annotation_helper.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file {pdf_path} not found.")
        sys.exit(1)
    
    extract_blocks_for_annotation(pdf_path)

if __name__ == "__main__":
    main() 