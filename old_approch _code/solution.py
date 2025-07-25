import fitz  # PyMuPDF
import json
import os
import re
import joblib  # To load the pre-trained model
import numpy as np

# --- Configuration ---
INPUT_DIR = "app\input"
OUTPUT_DIR = "app\output"
MODEL_PATH = "model.pkl"  # Your pre-trained classifier


def get_font_stats(doc):
    """
    Calculates statistics about font sizes used in the document.
    This is done once per document for efficiency.
    """
    sizes = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            size = round(span["size"])
                            sizes[size] = sizes.get(size, 0) + 1
    
    # Sort sizes by frequency
    sorted_sizes = sorted(sizes.keys(), reverse=True)
    # The most frequent size is likely the body text
    body_size = max(sizes, key=sizes.get)
    
    return sorted_sizes, body_size


def extract_features(line, page, doc_font_ranks, doc_body_size):
    """
    Extracts a feature vector for a single line of text.
    This is the core of the "intelligence".
    """
    # 1. Font-Based Features
    span = line["spans"][0] # Assume first span is representative
    font_size = round(span["size"])
    is_bold = "bold" in span["font"].lower()
    
    try:
        font_size_rank = doc_font_ranks.index(font_size) + 1
    except ValueError:
        font_size_rank = -1 # Should not happen if pre-calculated

    # 2. Positional Features
    page_width = page.rect.width
    line_width = line["bbox"][2] - line["bbox"][0]
    is_centered = abs(((line["bbox"][0] + line["bbox"][2]) / 2) - (page_width / 2)) < (page_width * 0.1) # Within 10% of center

    # 3. Text-Based Features
    text = "".join([s["text"] for s in line["spans"]]).strip()
    word_count = len(text.split())
    ends_with_period = text.endswith('.')
    is_all_caps = text.isupper() and len(text) > 1
    starts_with_numbering = bool(re.match(r'^\d+[\.\)]', text))

    # Feature vector - The order MUST match your training data
    features = [
        font_size,
        font_size_rank,
        int(is_bold),
        int(is_centered),
        word_count,
        int(ends_with_period),
        int(is_all_caps),
        int(starts_with_numbering)
    ]
    return np.array(features).reshape(1, -1), text


def process_pdf(pdf_path, model):
    """
    Processes a single PDF file to extract its outline.
    """
    doc = fitz.open(pdf_path)
    if doc.page_count > 50:
        print(f"Warning: {os.path.basename(pdf_path)} has more than 50 pages.")

    # Pre-calculate document-wide font statistics
    doc_font_ranks, doc_body_size = get_font_stats(doc)

    outline = []
    potential_title = ""
    highest_title_score = -1

    class_mapping = {1: "H3", 2: "H2", 3: "H1", 4: "Title"}

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if not line["spans"] or not "".join(s["text"] for s in line["spans"]).strip():
                        continue

                    features, text = extract_features(line, page, doc_font_ranks, doc_body_size)
                    
                    # Predict the class (0=Para, 1=H3, 2=H2, 3=H1, 4=Title)
                    pred_class = model.predict(features)[0]

                    if pred_class in class_mapping:
                        level = class_mapping[pred_class]
                        if level == "Title" and page_num < 2: # Title is usually on first 2 pages
                            # Simple logic: highest rank on early page is title
                            if features[0, 1] > highest_title_score:
                                potential_title = text
                                highest_title_score = features[0, 1]
                        else:
                            outline.append({"level": level, "text": text, "page": page_num + 1})

    return {"title": potential_title, "outline": outline}


if __name__ == "__main__":
    print("Starting PDF processing...")
    model = joblib.load(MODEL_PATH)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            print(f"Processing {filename}...")
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
            
            result = process_pdf(pdf_path, model)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Successfully generated {output_path}")

    print("Processing complete.")