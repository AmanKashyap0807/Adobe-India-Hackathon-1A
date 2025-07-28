import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageTk
import fitz  # PyMuPDF

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.features import extract_document_adaptive_features, is_document_artifact

# Add this helper function
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

class ComprehensivePDFAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Comprehensive PDF Text Block Annotation Tool - All Features Integrated")
        self.root.geometry("1400x1000")
        
        # Variables
        self.current_pdf = None
        self.current_blocks = []
        self.current_groups = []
        self.artifacts = []
        self.current_block_index = 0
        self.current_group_index = 0
        self.annotations = {}
        self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0, "Artifact": 0}
        self.artifact_count = 0
        self.show_artifacts = tk.BooleanVar(value=False)
        self.doc = None  # PyMuPDF document
        
        # Mode selection
        self.annotation_mode = tk.StringVar(value="enhanced")  # enhanced, basic, groups
        
        # Label mappings
        self.label_map = {
            "1": "H1",
            "2": "H2", 
            "3": "H3",
            "4": "Title",
            "0": "Body",
            "9": "Artifact"
        }
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frames
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for PDF preview and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for text blocks and annotation
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # === LEFT PANEL ===
        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="PDF Selection", padding="5")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Select PDF", command=self.load_pdf).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(left_frame, text="Annotation Mode", padding="5")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Enhanced (with artifacts)", variable=self.annotation_mode, 
                       value="enhanced", command=self.on_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Basic (legacy)", variable=self.annotation_mode, 
                       value="basic", command=self.on_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Group-based", variable=self.annotation_mode, 
                       value="groups", command=self.on_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        
        # PDF Preview
        preview_frame = ttk.LabelFrame(left_frame, text="PDF Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pdf_preview = ttk.Label(preview_frame, text="No PDF loaded")
        self.pdf_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Page navigation
        page_frame = ttk.Frame(left_frame)
        page_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(page_frame, text="Previous Page", command=self.previous_page).pack(side=tk.LEFT, padx=5, pady=5)
        self.page_label = ttk.Label(page_frame, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(page_frame, text="Next Page", command=self.next_page).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Artifact filtering controls
        artifact_frame = ttk.LabelFrame(left_frame, text="Artifact Handling", padding="5")
        artifact_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(
            artifact_frame, 
            text="Show Detected Artifacts (for review)", 
            variable=self.show_artifacts,
            command=self.toggle_artifacts
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        self.artifact_stats_label = ttk.Label(artifact_frame, text="Artifacts: 0/0 (0%)")
        self.artifact_stats_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # === RIGHT PANEL ===
        # Progress info
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0/0")
        self.progress_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Block info
        self.block_info_label = ttk.Label(progress_frame, text="Block Info: Single line")
        self.block_info_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Efficiency info (for group mode)
        self.efficiency_label = ttk.Label(progress_frame, text="")
        self.efficiency_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Annotation counts
        counts_frame = ttk.LabelFrame(right_frame, text="Annotation Counts", padding="5")
        counts_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.counts_label = ttk.Label(counts_frame, text="Title:0 H1:0 H2:0 H3:0 Body:0 Artifact:0")
        self.counts_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Text display with highlighting
        text_frame = ttk.LabelFrame(right_frame, text="Text Block Content", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_display = scrolledtext.ScrolledText(text_frame, height=15, width=60, wrap=tk.WORD, font=("Arial", 11))
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Individual lines preview (for group mode)
        lines_frame = ttk.LabelFrame(right_frame, text="Individual Lines Preview", padding="5")
        lines_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.lines_preview = scrolledtext.ScrolledText(lines_frame, height=4, width=60, wrap=tk.WORD, font=("Arial", 10))
        self.lines_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Annotation buttons
        button_frame = ttk.LabelFrame(right_frame, text="Annotation Controls", padding="5")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid of annotation buttons
        annotation_grid = ttk.Frame(button_frame)
        annotation_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(annotation_grid, text="1 - H1", command=lambda: self.annotate("1"), width=10).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(annotation_grid, text="2 - H2", command=lambda: self.annotate("2"), width=10).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(annotation_grid, text="3 - H3", command=lambda: self.annotate("3"), width=10).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(annotation_grid, text="4 - Title", command=lambda: self.annotate("4"), width=10).grid(row=0, column=3, padx=2, pady=2)
        ttk.Button(annotation_grid, text="0 - Body", command=lambda: self.annotate("0"), width=10).grid(row=0, column=4, padx=2, pady=2)
        ttk.Button(annotation_grid, text="9 - Artifact", command=lambda: self.annotate("9"), width=10, style="Artifact.TButton").grid(row=0, column=5, padx=2, pady=2)
        
        # Create a style for the artifact button
        style = ttk.Style()
        style.configure("Artifact.TButton", foreground="red")
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.previous_block).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(nav_frame, text="Next", command=self.next_block).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(nav_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(nav_frame, text="Create Template", command=self.create_annotation_template).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Keyboard bindings
        self.root.bind('<Key-1>', lambda e: self.annotate("1"))
        self.root.bind('<Key-2>', lambda e: self.annotate("2"))
        self.root.bind('<Key-3>', lambda e: self.annotate("3"))
        self.root.bind('<Key-4>', lambda e: self.annotate("4"))
        self.root.bind('<Key-0>', lambda e: self.annotate("0"))
        self.root.bind('<Key-9>', lambda e: self.annotate("9"))
        self.root.bind('<Left>', lambda e: self.previous_block())
        self.root.bind('<Right>', lambda e: self.next_block())
        self.root.bind('<space>', lambda e: self.next_block())
        self.root.focus_set()
        
    def on_mode_change(self):
        """Handle mode change"""
        if self.current_pdf:
            self.load_pdf()  # Reload with new mode
        
    def load_pdf(self):
        """Load a PDF file and extract text blocks based on selected mode"""
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not file_path:
            return
            
        self.current_pdf = file_path
        self.file_label.config(text=f"File: {Path(file_path).name}")
        
        # Close any previously opened document
        if self.doc:
            self.doc.close()
            
        # Open the PDF with PyMuPDF
        self.doc = fitz.open(file_path)
        self.current_page = 0
        
        # Extract text blocks based on mode
        if self.annotation_mode.get() == "enhanced":
            self.extract_enhanced_blocks()
        elif self.annotation_mode.get() == "basic":
            self.extract_basic_blocks()
        elif self.annotation_mode.get() == "groups":
            self.extract_grouped_blocks()
        
    def extract_enhanced_blocks(self):
        """Extract blocks with artifact pre-identification (Enhanced mode)"""
        try:
            # Extract features with artifact detection
            features_list, doc_stats = extract_document_adaptive_features(self.current_pdf)
            
            # Show filtering results to user
            self.show_filtering_stats(doc_stats)
            
            # Convert to annotation format
            self.current_blocks = []
            self.artifacts = []
            
            for i, features in enumerate(features_list):
                # Add semantic information
                semantic_info = ""
                if features.get('is_bullet_list', 0) == 1:
                    semantic_info = " [BULLET LIST]"
                elif features.get('is_table_content', 0) == 1:
                    semantic_info = " [TABLE CONTENT]"
                elif features.get('element_count', 1) > 1:
                    semantic_info = f" [MULTI-LINE: {features.get('element_count')} parts]"
                
                block_data = {
                    "id": i,
                    "text": features["text"],
                    "page": features["page_num"],
                    "line_count": features.get("line_count", 1),
                    "is_multiline": features.get("is_multiline", 0),
                    "is_artifact": False,  # These are content blocks
                    "bbox": features.get("bbox", [0, 0, 0, 0]),
                    "semantic_info": semantic_info,
                    "element_count": features.get("element_count", 1),
                    "semantic_type": features.get("semantic_type", "text_block")
                }
                self.current_blocks.append(block_data)
            
            # Reset annotation state
            self.current_block_index = 0
            self.annotations = {}
            self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0, "Artifact": 0}
            
            # Update artifact statistics
            artifact_count = doc_stats.get('artifact_lines', 0)
            total_lines = doc_stats.get('total_lines', 0)
            artifact_pct = (artifact_count / total_lines * 100) if total_lines > 0 else 0
            self.artifact_stats_label.config(text=f"Artifacts: {artifact_count}/{total_lines} ({artifact_pct:.1f}%)")
            
            # Render the first page
            self.render_current_page()
            
            # Display the first block
            self.display_current_block()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    
    def extract_basic_blocks(self):
        """Extract blocks using basic method (Legacy mode)"""
        try:
            # Import the updated feature extraction that groups lines
            features_list, doc_stats = extract_document_adaptive_features(self.current_pdf)
            
            # Convert features back to block format for annotation
            self.current_blocks = []
            for i, features in enumerate(features_list):
                # Check if this might be an artifact (using position)
                is_artifact = False
                y_position = features.get("y_position_normalized", 0)
                if y_position < 0.08 or y_position > 0.92:
                    is_artifact = True
                
                self.current_blocks.append({
                    "id": i,
                    "text": features["text"],
                    "page": features["page_num"] + 1,
                    "line_count": features.get("line_count", 1),
                    "is_multiline": features.get("is_multiline", 0),
                    "is_likely_artifact": is_artifact
                })
            
            # Show artifact statistics
            if 'artifact_percentage' in doc_stats:
                messagebox.showinfo(
                    "Artifact Detection", 
                    f"Document Statistics:\n\n"
                    f"Total lines: {doc_stats.get('total_lines', 0)}\n"
                    f"Content lines: {doc_stats.get('content_lines', 0)}\n"
                    f"Artifact lines: {doc_stats.get('artifact_lines', 0)}\n"
                    f"Artifact percentage: {doc_stats.get('artifact_percentage', 0):.1f}%\n\n"
                    f"Artifacts have been pre-filtered from document statistics.\n"
                    f"You can still mark remaining blocks as artifacts using '9'."
                )
            
            self.current_block_index = 0
            self.annotations = {}
            self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0, "Artifact": 0}
            self.display_current_block()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    
    def extract_grouped_blocks(self):
        """Extract grouped text blocks using the Two-Stage Hybrid Approach (Group mode)"""
        try:
            # Use the updated feature extraction that groups lines
            features_list, doc_stats = extract_document_adaptive_features(self.current_pdf)
            
            # Convert features back to group format for annotation
            self.current_groups = []
            total_lines = 0
            
            for i, features in enumerate(features_list):
                group_info = {
                    "group_id": i,
                    "text": features["text"],
                    "page": features["page_num"] + 1,
                    "line_count": features.get("line_count", 1),
                    "is_multiline": features.get("is_multiline", 0),
                    "line_font_consistency": features.get("line_font_consistency", 1.0),
                    "original_lines": features.get("original_lines", [])
                }
                self.current_groups.append(group_info)
                total_lines += group_info["line_count"]
            
            self.current_group_index = 0
            self.annotations = {}
            self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0}
            
            # Update efficiency statistics
            speed_gain = total_lines / len(self.current_groups) if self.current_groups else 0
            self.efficiency_label.config(
                text=f"Groups: {len(self.current_groups)} | Lines: {total_lines} | Gain: {speed_gain:.1f}x"
            )
            
            self.display_current_group()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    
    def show_filtering_stats(self, doc_stats):
        """Show filtering statistics to user"""
        artifact_count = doc_stats.get('artifact_lines', 0)
        total_lines = doc_stats.get('total_lines', 0)
        content_lines = doc_stats.get('content_lines', 0)
        
        artifact_pct = (artifact_count / total_lines * 100) if total_lines > 0 else 0
        
        stats_msg = f"Preprocessing Results:\n\n"
        stats_msg += f"Content blocks: {content_lines}\n"
        stats_msg += f"Artifacts filtered: {artifact_count}\n"
        stats_msg += f"Total blocks: {total_lines}\n\n"
        stats_msg += f"Artifact percentage: {artifact_pct:.1f}%\n\n"
        stats_msg += f"Artifacts have been automatically filtered.\n"
        stats_msg += f"You can review them by checking 'Show Detected Artifacts'."
        
        messagebox.showinfo("Artifact Detection Results", stats_msg)
    
    def toggle_artifacts(self):
        """Toggle showing artifacts for review"""
        if not self.current_pdf:
            return
            
        # Re-extract blocks with artifacts if needed
        if self.show_artifacts.get():
            messagebox.showinfo(
                "Artifact Review Mode", 
                "You are now in artifact review mode.\n\n"
                "Artifacts are shown with a red border in the preview.\n"
                "You can verify and change artifact detection if needed."
            )
        else:
            messagebox.showinfo(
                "Normal Mode", 
                "Returned to normal annotation mode.\n"
                "Artifacts are hidden for a cleaner annotation experience."
            )
            
        # Refresh the display
        self.render_current_page()
        if self.annotation_mode.get() == "groups":
            self.display_current_group()
        else:
            self.display_current_block()
    
    def render_current_page(self):
        """Render the current page with block highlighting"""
        if not self.doc or self.current_page >= len(self.doc):
            return
            
        page = self.doc[self.current_page]
        
        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Scale factor for display (adjust as needed)
        scale = min(500 / page_width, 700 / page_height)
        
        # Render page to pixmap
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Create a drawing context
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Highlight blocks on the current page
        if self.annotation_mode.get() == "groups":
            blocks_to_highlight = self.current_groups
        else:
            blocks_to_highlight = self.current_blocks
            
        for block in blocks_to_highlight:
            if block["page"] == self.current_page + 1:  # Adjust for 1-indexed pages
                # Get block position
                bbox = block.get("bbox", [0, 0, 0, 0])
                x0, y0, x1, y1 = [coord * scale for coord in bbox]
                
                # Determine color based on annotation or current position
                color = "blue"  # Default
                block_id = block.get("id", block.get("group_id"))
                if block_id in self.annotations:
                    label = self.annotations[block_id]
                    if label == "Title":
                        color = "purple"
                    elif label.startswith("H"):
                        color = "green"
                    elif label == "Body":
                        color = "blue"
                    elif label == "Artifact":
                        color = "red"
                
                # Make current block highlight brighter
                current_index = self.current_block_index if self.annotation_mode.get() != "groups" else self.current_group_index
                current_blocks = self.current_blocks if self.annotation_mode.get() != "groups" else self.current_groups
                if current_blocks and current_index < len(current_blocks):
                    current_block = current_blocks[current_index]
                    current_id = current_block.get("id", current_block.get("group_id"))
                    if block_id == current_id:
                        color = "yellow"
                
                # Draw rectangle
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Also highlight artifacts if showing them
        if self.show_artifacts.get() and self.artifacts:
            for artifact in self.artifacts:
                if artifact["page"] == self.current_page:
                    bbox = artifact.get("bbox", [0, 0, 0, 0])
                    x0, y0, x1, y1 = [coord * scale for coord in bbox]
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Update label
        self.pdf_preview.config(image=photo)
        self.pdf_preview.image = photo  # Keep a reference
        
        # Update page label
        self.page_label.config(text=f"Page: {self.current_page + 1}/{len(self.doc)}")
    
    def display_current_block(self):
        """Enhanced display with artifact indication"""
        if not self.current_blocks:
            return
            
        block = self.current_blocks[self.current_block_index]
        
        # Update text display
        self.text_display.delete(1.0, tk.END)
        
        # Format text based on block type
        self.text_display.tag_configure("artifact", foreground="red")
        self.text_display.tag_configure("title", font=("Arial", 12, "bold"))
        self.text_display.tag_configure("header", foreground="blue")
        self.text_display.tag_configure("multiline", background="#f0f0f0")
        
        # Insert page info with semantic information
        semantic_info = block.get('semantic_info', '')
        self.text_display.insert(tk.END, f"Page {block['page'] + 1}{semantic_info}\n\n")
        
        # Add artifact indicator if needed
        if block.get('is_artifact', False):
            self.text_display.insert(tk.END, "[ARTIFACT - Please verify]\n\n", "artifact")
        
        # If this block has been annotated, show the label
        if block["id"] in self.annotations:
            label = self.annotations[block["id"]]
            self.text_display.insert(tk.END, f"[{label}]\n\n", "header")
        
        # Insert the actual text
        tag = None
        if block.get('is_multiline', False):
            tag = "multiline"
        self.text_display.insert(tk.END, block['text'], tag)
        
        # Update progress
        total = len(self.current_blocks)
        current = self.current_block_index + 1
        self.progress_label.config(text=f"Progress: {current}/{total}")
        
        # Update block info with semantic details
        line_info = f"Lines: {block.get('line_count', 1)}"
        if block.get('is_multiline', False):
            line_info += " (MULTI-LINE BLOCK)"
        
        semantic_type = block.get('semantic_type', 'text_block')
        element_count = block.get('element_count', 1)
        line_info += f", Type: {semantic_type.title()}, Elements: {element_count}"
        
        self.block_info_label.config(text=f"Block Info: {line_info}")
        
        # Update counts
        counts_text = " ".join([f"{k}:{v}" for k, v in self.annotation_counts.items()])
        self.counts_label.config(text=counts_text)
        
        # Ensure current page is showing
        if self.current_page != block["page"]:
            self.current_page = block["page"]
            self.render_current_page()
        else:
            # Just update highlighting on current page
            self.render_current_page()
    
    def display_current_group(self):
        """Display current group for group-based annotation"""
        if not self.current_groups:
            return
            
        group = self.current_groups[self.current_group_index]
        
        # Display main text
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, f"Page {group['page']}\n\n{group['text']}")
        
        # Update progress
        total = len(self.current_groups)
        current = self.current_group_index + 1
        self.progress_label.config(text=f"Progress: {current}/{total}")
        
        # Update group information
        group_type = "Multi-line" if group['is_multiline'] else "Single-line"
        self.block_info_label.config(
            text=f"Group: {current}/{total} | Lines: {group['line_count']} | Type: {group_type} | Font Consistency: {group['line_font_consistency']:.2f}"
        )
        
        # Show individual lines preview
        self.lines_preview.delete(1.0, tk.END)
        if group.get('original_lines'):
            lines_text = "Individual lines in this group:\n"
            for j, line in enumerate(group['original_lines']):
                lines_text += f"{j+1}. {line['text']}\n"
            self.lines_preview.insert(1.0, lines_text)
        else:
            self.lines_preview.insert(1.0, "Single line group")
        
        # Update counts
        counts_text = " | ".join([f"{k}:{v}" for k, v in self.annotation_counts.items()])
        self.counts_label.config(text=counts_text)
        
        # Ensure current page is showing
        if self.current_page != group["page"] - 1:  # Convert to 0-indexed
            self.current_page = group["page"] - 1
            self.render_current_page()
        else:
            # Just update highlighting on current page
            self.render_current_page()
    
    def previous_page(self):
        """Go to previous page"""
        if not self.doc or self.current_page <= 0:
            return
            
        self.current_page -= 1
        self.render_current_page()
    
    def next_page(self):
        """Go to next page"""
        if not self.doc or self.current_page >= len(self.doc) - 1:
            return
            
        self.current_page += 1
        self.render_current_page()
    
    def annotate(self, key):
        """Annotate current block with specified label"""
        if self.annotation_mode.get() == "groups":
            if not self.current_groups:
                return
                
            label = self.label_map[key]
            group_id = self.current_groups[self.current_group_index]["group_id"]
            
            # Remove old annotation if exists
            if group_id in self.annotations:
                old_label = self.annotations[group_id]
                self.annotation_counts[old_label] -= 1
                
            # Add new annotation
            self.annotations[group_id] = label
            self.annotation_counts[label] += 1
            
            # Auto-advance to next group
            self.next_block()
        else:
            if not self.current_blocks:
                return
                
            label = self.label_map[key]
            block_id = self.current_blocks[self.current_block_index]["id"]
            
            # Remove old annotation if exists
            if block_id in self.annotations:
                old_label = self.annotations[block_id]
                self.annotation_counts[old_label] -= 1
                
            # Add new annotation
            self.annotations[block_id] = label
            self.annotation_counts[label] += 1
            
            # Auto-advance to next block
            self.next_block()
        
    def previous_block(self):
        """Go to previous block/group"""
        if self.annotation_mode.get() == "groups":
            if not self.current_groups or self.current_group_index <= 0:
                return
                
            self.current_group_index -= 1
            self.display_current_group()
        else:
            if not self.current_blocks or self.current_block_index <= 0:
                return
                
            self.current_block_index -= 1
            self.display_current_block()
            
    def next_block(self):
        """Go to next block/group"""
        if self.annotation_mode.get() == "groups":
            if not self.current_groups or self.current_group_index >= len(self.current_groups) - 1:
                return
                
            self.current_group_index += 1
            self.display_current_group()
        else:
            if not self.current_blocks or self.current_block_index >= len(self.current_blocks) - 1:
                return
                
            self.current_block_index += 1
            self.display_current_block()
    
    def create_annotation_template(self):
        """Create annotation template from PDF (from annotation_helper.py)"""
        if not self.current_pdf:
            messagebox.showwarning("Warning", "No PDF loaded")
            return
            
        try:
            # Extract blocks using basic method
            features_list, _ = extract_document_adaptive_features(self.current_pdf)
            
            # Create annotation template
            annotation_data = {
                "pdf_file": os.path.basename(self.current_pdf),
                "blocks": []
            }
            
            for i, features in enumerate(features_list):
                text = features.get("text", "").strip()
                if text:  # Only include non-empty blocks
                    annotation_data["blocks"].append({
                        "id": i,
                        "text": text,
                        "page": features.get("page_num", 0) + 1,
                        "role": "Body"  # Default role, to be changed during annotation
                    })
            
            # Save annotation template
            pdf_name = Path(self.current_pdf).stem
            output_file = f"data/annotated/{pdf_name}_template.json"
            
            ensure_dir("data/annotated")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo(
                "Template Created", 
                f"Annotation template created: {output_file}\n"
                f"Found {len(annotation_data['blocks'])} text blocks\n\n"
                f"To annotate:\n"
                f"1. Open the JSON file in any text editor\n"
                f"2. Change 'role' values to: Title, H1, H2, H3, or Body\n"
                f"3. Save the file"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create template: {str(e)}")
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        if not self.current_pdf:
            messagebox.showwarning("Warning", "No PDF loaded")
            return
            
        # Create directory if it doesn't exist
        ensure_dir("data")
        ensure_dir("data/annotated")
        
        # Prepare annotation data based on mode
        pdf_name = Path(self.current_pdf).stem
        
        if self.annotation_mode.get() == "groups":
            # Group-based format
            annotation_data = {
                "pdf_file": self.current_pdf,
                "total_groups": len(self.current_groups),
                "total_lines": sum(g['line_count'] for g in self.current_groups),
                "multi_line_groups": sum(1 for g in self.current_groups if g['is_multiline']),
                "annotation_counts": self.annotation_counts,
                "efficiency_gain": sum(g['line_count'] for g in self.current_groups) / len(self.current_groups),
                "groups": []
            }
            
            # Add group annotations
            for group in self.current_groups:
                group_data = {
                    "group_id": group["group_id"],
                    "text": group["text"],
                    "page": group["page"],
                    "line_count": group["line_count"],
                    "is_multiline": group["is_multiline"],
                    "line_font_consistency": group["line_font_consistency"],
                    "role": self.annotations.get(group["group_id"], "Body"),
                    "individual_lines": []
                }
                
                # Add individual line information for backend processing
                if group.get('original_lines'):
                    for line in group['original_lines']:
                        group_data["individual_lines"].append({
                            "text": line['text'],
                            "bbox": line['bbox'],
                            "font_size": line['font_size'],
                            "page_num": line['page_num']
                        })
                
                annotation_data["groups"].append(group_data)
        else:
            # Standard format
            annotation_data = {
                "pdf_path": self.current_pdf,
                "total_blocks": len(self.current_blocks),
                "annotated_blocks": len(self.annotations),
                "multi_line_blocks": sum(1 for b in self.current_blocks if b.get('is_multiline', False)),
                "artifact_blocks": self.annotation_counts["Artifact"],
                "blocks": []
            }
            
            # Add block data
            for block in self.current_blocks:
                block_data = {
                    "id": block["id"],
                    "text": block["text"],
                    "page": block["page"] + 1,  # Convert to 1-indexed
                    "line_count": block.get("line_count", 1),
                    "is_multiline": block.get("is_multiline", False),
                    "is_artifact": self.annotations.get(block["id"], "Body") == "Artifact",
                    "role": self.annotations.get(block["id"], "Body")
                }
                annotation_data["blocks"].append(block_data)
        
        # Save to file
        output_file = f"data/annotated/{pdf_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
        messagebox.showinfo(
            "Success", 
            f"Annotations saved to {output_file}\n\n"
            f"Mode: {self.annotation_mode.get()}\n"
            f"Total blocks/groups: {len(self.current_blocks) if self.annotation_mode.get() != 'groups' else len(self.current_groups)}\n"
            f"Annotated blocks/groups: {len(self.annotations)}"
        )

def main():
    root = tk.Tk()
    app = ComprehensivePDFAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
