import fitz                          # PyMuPDF
import numpy as np
import re, unicodedata, logging, itertools
from collections import defaultdict
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#  ████  STRING CLEAN-UP
# ----------------------------------------------------------------------
TRANSLATE_MAP = {
    "\uf0b7": "•",  "\uf0a7": "•",  "\uf020": " ",
    "\uf04d": "-",  "\uf02d": "-",  "\u2022": "•",
}

def clean_special_characters(text: str) -> str:
    for bad, good in TRANSLATE_MAP.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t"
    )
    return text.strip()

# ----------------------------------------------------------------------
#  ████  ARTIFACT / WATERMARK / FOOTNOTE  FILTER
# ----------------------------------------------------------------------
HEADER_FOOTER_PAT = [
    r"^\s*\d+\s*$",                           # bare page number
    r"^\s*page\s+\d+", r"\d+\s+of\s+\d+",     # “Page 3”, “3 of 12”
    r"©.*\d{4}", r"www\.", r"http", r"^_+$",  # copyright / url / lines
]
HEADER_FOOTER_RE = [re.compile(p, re.I) for p in HEADER_FOOTER_PAT]

def is_document_artifact(block, h, w) -> bool:
    txt = block["text"].strip()
    y0 = block["bbox"][1] / h
    if any(rx.match(txt) for rx in HEADER_FOOTER_RE):
        return True
    if y0 < 0.08 or y0 > 0.92:                     # header/footer band
        return True
    if len(txt) < 5 and txt.isalpha() is False:    # tiny noise
        return True
    return False

# ----------------------------------------------------------------------
#  ████  TABLE DETECTION  (bordered / border-less / multi-page / merged)
# ----------------------------------------------------------------------
def format_table_as_text(table):
    return "\n".join(" | ".join(cell or "" for cell in row) for row in table)

def _merge_table_cells(table):
    """Re-expand empty cells created by rowspan / colspan merges."""
    out = []
    for row in table:
        new = []
        for cell in row:
            if cell is None or str(cell).strip() == "":
                new.append("")
            else:
                new.append(str(cell).strip())
        out.append(new)
    return out

def detect_tables(page):
    if hasattr(page, "find_tables"):
        for tbl in page.find_tables():
            table = tbl.extract()
            if not table:
                continue
            yield {
                "bbox": tbl.bbox,
                "text": format_table_as_text(_merge_table_cells(table)),
                "rows": len(table),
                "cols": max(len(r) for r in table),
            }
    # ---------- border-less fallback (alignment clustering) ----------
    blocks = [
        span for blk in page.get_text("dict")["blocks"] if blk["type"] == 0
        for line in blk["lines"] for span in line["spans"] if span["text"].strip()
    ]
    rows = defaultdict(list)
    for sp in blocks:
        rows[int(sp["bbox"][1]//4)].append(sp)      # 4-pt row bin
    for rkey, spans in rows.items():
        spans.sort(key=lambda s: s["bbox"][0])
        gaps = np.diff([s["bbox"][0] for s in spans])
        if len(spans) >= 3 and (gaps.std() / (gaps.mean()+1e-4) < .5):
            # likely a table row
            text = " | ".join(clean_special_characters(s["text"]) for s in spans)
            bbox = (
                spans[0]["bbox"][0], min(s["bbox"][1] for s in spans),
                spans[-1]["bbox"][2], max(s["bbox"][3] for s in spans)
            )
            yield {"bbox": bbox, "text": text, "rows": 1, "cols": len(spans)}

# ----------------------------------------------------------------------
#  ████  BULLET  DETECTION   (vector, glyph, inline-img, nested)
# ----------------------------------------------------------------------
def detect_vector_bullets(page):
    bullets = []
    for d in page.get_drawings():
        r = d.get("rect");     t = d.get("type")
        if not r or t not in ("f", "fs"):           # filled shape only
            continue
        if max(r.width, r.height) <= 14:            # small dot / square
            bullets.append([r.x0, r.y0, r.x1, r.y1])
    return bullets

GLYPH_BULLETS = {"•", "▪", "■", "●", "◦", "‣", "▸", "▹", "-"}

def associate_bullets(page, spans, bullets):
    """Attach detected bullets to nearest text span start."""
    if not spans:  # If there are no text spans, do nothing
        return []
        
    span_objs = sorted(spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    
    # Initialize bullet attributes for all spans
    for s in span_objs:
        s["is_bullet"] = False
        s["nest_lvl"]  = 0
    
    # vector bullets ---------------------------------------------------
    for b in bullets:
        cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
        # nearest span above-right
        if span_objs:  # Check if there are any spans
            tgt = min(span_objs, key=lambda s:
                    abs(s["bbox"][1]-cy) + abs(s["bbox"][0]-cx))
            tgt["is_bullet"] = True
    
    # glyph bullets ----------------------------------------------------
    for sp in span_objs:
        leading = sp["text"].lstrip()[:2]
        if leading and leading[0] in GLYPH_BULLETS:
            sp["text"] = sp["text"].lstrip()[1:].lstrip()
            sp["is_bullet"] = True
    
    # nesting level (indent) ------------------------------------------
    indents = sorted({int(s["bbox"][0]//10)*10 for s in span_objs if s["is_bullet"]})
    for sp in span_objs:
        if sp["is_bullet"]:
            sp["nest_lvl"] = indents.index(int(sp["bbox"][0]//10)*10) if indents else 0
    
    return span_objs

# ----------------------------------------------------------------------
#  ████  SEMANTIC GROUPER  (column / page-break aware)
# ----------------------------------------------------------------------
def should_merge(a,b):
    # same page & column
    if a["page"] != b["page"]:
        return False
    col_w = a["page_width"]/2
    col_a = int(a["bbox"][0]//col_w)
    col_b = int(b["bbox"][0]//col_w)
    if col_a != col_b:
        return False
    # font similarity
    if abs(a["font_size"]-b["font_size"]) > 2: 
        return False
    if a["is_bold"]!=b["is_bold"] or a["is_italic"]!=b["is_italic"]:
        return False
    # vertical proximity
    return (b["bbox"][1]-a["bbox"][3]) < 50

def merge_spans(spans):
    spans = sorted(spans,key=lambda s:(s["page"],s["bbox"][1],s["bbox"][0]))
    groups=[]; cur=[spans[0]]
    for sp in spans[1:]:
        if should_merge(cur[-1], sp):
            cur.append(sp)
        else:
            groups.append(cur); cur=[sp]
    groups.append(cur)
    return groups

# ----------------------------------------------------------------------
#  ████  MASTER FEATURE EXTRACTOR  (extract_document_adaptive_features)
# ----------------------------------------------------------------------
def extract_document_adaptive_features(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans=[]
    for pno, page in enumerate(doc):
        p_h, p_w = page.rect.height, page.rect.width
        spans=[{
            "text": clean_special_characters(s["text"]),
            "page": pno,
            "bbox": s["bbox"],
            "font_size": s["size"],
            "is_bold": bool(s.get("flags",0)&16 or "Bold" in s["font"]),
            "is_italic": bool(s.get("flags",0)&2  or "Italic" in s["font"]),
            "page_height": p_h,
            "page_width":  p_w,
            "is_bullet": False,  # Initialize bullet attribute
            "nest_lvl": 0        # Initialize nesting level
        } for b in page.get_text("dict")["blocks"] if b["type"]==0
              for l in b["lines"] for s in l["spans"] if s["text"].strip()]
        
        # Detect and associate bullets with spans
        vector_bullets = detect_vector_bullets(page)
        spans = associate_bullets(page, spans, vector_bullets)
        all_spans.extend(spans)
        
        # ------- add table blocks --------------------------------------------------
        for tbl in detect_tables(page):
            all_spans.append({
                "text": tbl["text"], "page": pno, "bbox": tbl["bbox"],
                "font_size": 12, "is_bold": False, "is_italic": False,
                "page_height": p_h, "page_width": p_w,
                "semantic_type": "table_block", "is_table": 1,
                "is_bullet": False, "nest_lvl": 0  # Initialize bullet attributes for tables
            })
    doc.close()

    # filter artifacts
    clean = [
        s for s in all_spans
        if not is_document_artifact(s, s["page_height"], s["page_width"])
    ]

    # group
    groups=[]
    for g in merge_spans(clean):
        bbox = [
            min(s["bbox"][0] for s in g), min(s["bbox"][1] for s in g),
            max(s["bbox"][2] for s in g), max(s["bbox"][3] for s in g)
        ]
        txt  = " ".join(s["text"] for s in g)
        fs   = np.mean([s["font_size"] for s in g])
        groups.append({
            "text": txt, "page_num": g[0]["page"], "bbox": bbox,
            "font_size": fs, "is_bold": any(s["is_bold"] for s in g),
            "is_italic": any(s["is_italic"] for s in g),
            "page_height": g[0]["page_height"], "page_width": g[0]["page_width"],
            "element_count": len(g),
            "semantic_type": g[0].get("semantic_type","text_block"),
            "is_bullet_list": int(any(s.get("is_bullet", False) for s in g)),  # Safely access is_bullet
            "bullet_nesting": max((s.get("nest_lvl", 0) for s in g), default=0),  # Safely access nest_lvl
            "is_table_content": int(any(s.get("is_table", 0) for s in g)),
        })
    
    # ---------- compute document-level stats ---------------------------
    fsizes=[g["font_size"] for g in groups]
    tlens =[len(g["text"]) for g in groups]
    wcnts =[len(g["text"].split()) for g in groups]
    stats = {
        "fs_mean":np.mean(fsizes), "fs_std":np.std(fsizes)+1e-4,
        "fs_max": max(fsizes),     "fs_med":np.median(fsizes),
        "tl_mean":np.mean(tlens),  "tl_std":np.std(tlens)+1e-4,
        "wc_mean":np.mean(wcnts),  "wc_std":np.std(wcnts)+1e-4,
    }

    features=[]
    for i,g in enumerate(groups):
        f={
            "block_id": i, "text": g["text"], "page_num": g["page_num"],
            "font_size_zscore": (g["font_size"]-stats["fs_mean"])/stats["fs_std"],
            "font_size_percentile": (g["font_size"]-stats["fs_med"])/(stats["fs_max"]-stats["fs_med"]+1e-4),
            "font_size_ratio_max": g["font_size"]/stats["fs_max"],
            "font_size_ratio_median": g["font_size"]/stats["fs_med"],
            "is_bold": int(g["is_bold"]), "is_italic": int(g["is_italic"]),
            "y_position_normalized": g["bbox"][1]/g["page_height"],
            "x_position_normalized": g["bbox"][0]/g["page_width"],
            "space_above_ratio": 0.1,     # placeholder (needs page context)
            "horizontal_alignment": int(g["bbox"][0] < 0.1*g["page_width"]),
            "text_length_zscore": (len(g["text"])-stats["tl_mean"])/stats["tl_std"],
            "word_count_zscore": (len(g["text"].split())-stats["wc_mean"])/stats["wc_std"],
            "starts_with_number": int(bool(re.match(r"^[0-9IVXivx]+", g["text"].lstrip()))),
            "case_pattern": 3 if g["text"].isupper() else 2 if g["text"].istitle()
                            else 1 if g["text"].islower() else 0,
            "text_density_around": 0.5,   # simplified placeholder
            "follows_whitespace": 1,
            "text_uniqueness": 0.8,
            "semantic_type": g["semantic_type"],
            "element_count": g["element_count"],
            "is_bullet_list": g["is_bullet_list"],
            "is_table_content": g["is_table_content"],
            "bbox_aspect_ratio": (
                (g["bbox"][2]-g["bbox"][0])/
                max((g["bbox"][3]-g["bbox"][1]),1)
            ),
        }
        features.append(f)
    return features, stats
