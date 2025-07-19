import fitz  # PyMuPDF
import os
import re
import pandas as pd

# --- Configuration ---
TRAINING_PDF_DIR = "training_pdfs"
OUTPUT_CSV_PATH = "training_data.csv"


def get_font_stats(doc):
    """Calculates statistics about font sizes used in the document."""
    sizes = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            size = round(span["size"])
                            sizes[size] = sizes.get(size, 0) + 1
    
    if not sizes:
        return [], 0

    sorted_sizes = sorted(sizes.keys(), reverse=True)
    body_size = max(sizes, key=sizes.get)
    return sorted_sizes, body_size


def extract_features(line, page, doc_font_ranks):
    """Extracts a feature vector for a single line of text."""
    span = line["spans"][0]
    font_size = round(span["size"])
    is_bold = "bold" in span["font"].lower()
    
    try:
        font_size_rank = doc_font_ranks.index(font_size) + 1
    except ValueError:
        font_size_rank = -1

    page_width = page.rect.width
    is_centered = abs(((line["bbox"][0] + line["bbox"][2]) / 2) - (page_width / 2)) < (page_width * 0.1)

    text = "".join([s["text"] for s in line["spans"]]).strip()
    word_count = len(text.split())
    ends_with_period = text.endswith('.')
    is_all_caps = text.isupper() and len(text) > 1
    starts_with_numbering = bool(re.match(r'^\d+[\.\)]', text))

    features = [
        font_size, font_size_rank, int(is_bold), int(is_centered),
        word_count, int(ends_with_period), int(is_all_caps), int(starts_with_numbering)
    ]
    return features, text


def label_data():
    """Interactive script to label lines from PDFs."""
    all_features = []
    
    print("--- Starting Data Labeling ---")
    print("For each line, enter the corresponding number:")
    print("  0: Paragraph / Other")
    print("  1: H3 (Sub-sub-heading)")
    print("  2: H2 (Sub-heading)")
    print("  3: H1 (Main Heading)")
    print("  4: Title")
    print("  's': Skip line")
    print("  'q': Quit and save")
    print("-" * 20)

    for filename in os.listdir(TRAINING_PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        
        pdf_path = os.path.join(TRAINING_PDF_DIR, filename)
        print(f"\nProcessing Document: {filename}\n")
        doc = fitz.open(pdf_path)
        doc_font_ranks, _ = get_font_stats(doc)

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if not line["spans"] or not "".join(s["text"] for s in line["spans"]).strip():
                            continue
                        
                        features, text = extract_features(line, page, doc_font_ranks)
                        
                        user_input = input(f"Line: '{text}'\nLabel -> ")
                        
                        if user_input.lower() == 'q':
                            print("Quitting and saving...")
                            return all_features
                        if user_input.lower() == 's':
                            continue
                        if user_input in ['0', '1', '2', '3', '4']:
                            label = int(user_input)
                            all_features.append(features + [label])
    
    return all_features

if __name__ == "__main__":
    labeled_features = label_data()
    if labeled_features:
        columns = ['font_size', 'font_size_rank', 'is_bold', 'is_centered', 'word_count', 
                   'ends_with_period', 'is_all_caps', 'starts_with_numbering', 'label']
        df = pd.DataFrame(labeled_features, columns=columns)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully saved {len(df)} labeled samples to {OUTPUT_CSV_PATH}")