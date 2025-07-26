from pydantic import BaseModel, Field
from typing import List, Tuple

class TextBlock(BaseModel):
    """Represents a raw text block extracted from a PDF."""
    page_num: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    text: str

class EnhancedTextBlock(TextBlock):
    """
    Represents a text block after applying structural analysis.
    It includes a unique ID and a semantic tag.
    """
    block_id: int
    tag: str = Field(default="paragraph", description="The semantic tag of the block (e.g., heading, list_item).")

class DocumentStructure(BaseModel):
    """Represents the full, structured document."""
    file_name: str
    blocks: List[EnhancedTextBlock]

