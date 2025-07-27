import json
import lightgbm as lgb
import sys
import pathlib
import os

# Add the extract module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'extract'))
from features import pdf_to_blocks, compute_doc_stats, block_features

def load_model(model_path="model.txt"):
    """Load the trained LightGBM model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")
    return lgb.Booster(model_file=model_path)

def classify(pdf_path, model_path="model.txt"):
    """Classify text blocks in a PDF and generate outline."""
    clf = load_model(model_path)
    
    blocks = list(pdf_to_blocks(pdf_path))
    if not blocks:
        return {"title": "", "outline": []}
    
    stats = compute_doc_stats(blocks)
    page_h, page_w = 842, 595
    
    feats = [block_features(b, stats, page_h, page_w) for _, b in blocks]
    X = [[f[k] for k in ("font_ratio", "font_z", "word_count", "is_bold",
                         "space_above_ratio", "y_norm", "starts_number", "all_caps")]
         for f in feats]
    
    preds = clf.predict(X).argmax(1)
    level_map = {0: "Body", 1: "Title", 2: "H1", 3: "H2", 4: "H3"}
    
    outline = []
    for (page, blk), p in zip(blocks, preds):
        role = level_map[p]
        if role == "Body": 
            continue
        outline.append({
            "level": role, 
            "text": feats[len(outline)]["text"], 
            "page": page + 1
        })
    
    title = next((o for o in outline if o["level"] == "Title"), {"text": ""})["text"]
    return {"title": title, "outline": outline}

def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = pathlib.Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file {pdf_path} not found.")
        sys.exit(1)
    
    try:
        result = classify(str(pdf_path))
        output_file = f"{pdf_path.stem}.json"
        json.dump(result, open(output_file, "w"), indent=2)
        print(f"Results saved to {output_file}")
        print(f"Title: {result['title']}")
        print(f"Found {len(result['outline'])} headings")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 