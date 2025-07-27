import fitz
import numpy as np
import json
import os
import re

STAT_KEYS = ("median_font", "mean_font", "std_font", "max_font")

def pdf_to_blocks(pdf_path):
    """Extract text blocks from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:       # text only
                yield page_num, block
    doc.close()

def compute_doc_stats(blocks):
    """Compute font size statistics for the document."""
    sizes = [sp["size"] for _, b in blocks for l in b["lines"] for sp in l["spans"]]
    return dict(zip(STAT_KEYS,
        (np.median(sizes), np.mean(sizes), np.std(sizes), max(sizes))))

def block_features(block, stats, page_h, page_w):
    """Extract features from a text block for classification."""
    text = " ".join(sp["text"] for l in block["lines"] for sp in l["spans"]).strip()
    font_size = block["lines"][0]["spans"][0]["size"]
    y0 = block["bbox"][1]
    space_above = block["bbox"][1] - block["bbox_prev"][3] if "bbox_prev" in block else 0
    
    return {
       "font_ratio": font_size / stats["median_font"],
       "font_z": (font_size - stats["mean_font"]) / stats["std_font"],
       "word_count": len(text.split()),
       "is_bold": int("Bold" in block["lines"][0]["spans"][0]["font"]),
       "space_above_ratio": space_above / stats["median_font"],
       "y_norm": y0 / page_h,
       "starts_number": int(bool(re.match(r'^\d+([.]\d+)*', text))),
       "all_caps": int(text.isupper()),
       "text": text        # kept only for debugging / JSON
    } 