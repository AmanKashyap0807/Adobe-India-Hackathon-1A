import pandas as pd
import glob
import json
import os
import sys

# Add the extract module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'extract'))
from features import pdf_to_blocks, compute_doc_stats, block_features

def build_dataset():
    """Build training dataset from annotated PDFs."""
    rows, labels = [], []
    
    for pdf in glob.glob("data/raw/*.pdf"):
        pdf_name = os.path.basename(pdf)
        annotation_file = f"data/annotated/{pdf_name}.json"
        
        # Skip if no annotation file exists
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation file found for {pdf_name}")
            continue
            
        ann = json.load(open(annotation_file))
        labelled = {x["id"]: x["role"] for x in ann["blocks"]}
        blocks = list(pdf_to_blocks(pdf))
        stats = compute_doc_stats(blocks)
        page_h, page_w = 842, 595  # A4 defaults; PyMuPDF gives per page too
        
        for pid, blk in blocks:
            feats = block_features(blk, stats, page_h, page_w)
            rows.append(feats)
            labels.append(labelled.get(blk["id"], "Body"))
    
    df = pd.DataFrame(rows)
    df["label"] = labels
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset created with {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    build_dataset() 