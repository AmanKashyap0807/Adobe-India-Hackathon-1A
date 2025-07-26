import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts all text blocks from a PDF file along with their metadata.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A list of dictionaries, where each dictionary represents a
        text block and contains its content and metadata.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in enumerate(doc):
        # The 'dict' option provides a detailed structure of the page content.
        # We are interested in the 'blocks' which contain text.
        page_content = page.get_text("dict")
        
        for block in page_content.get("blocks", []):
            if block['type'] == 0:  # 0 indicates a text block
                # We can extract more details from spans if needed
                # For now, we'll take the first span's font info as representative
                if block.get("lines"):
                    first_line = block["lines"][0]
                    if first_line.get("spans"):
                        first_span = first_line["spans"][0]
                        block_info = {
                            "page_num": page_num,
                            "bbox": block["bbox"],
                            "font_size": first_span["size"],
                            "font_name": first_span["font"],
                            "text": "\n".join(["".join([span["text"] for span in line["spans"]]) for line in block["lines"]]).strip()
                        }
                        all_blocks.append(block_info)
    
    doc.close()
    return all_blocks

