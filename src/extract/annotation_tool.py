import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from pathlib import Path
import fitz  # PyMuPDF

class PDFAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Text Block Annotation Tool - 20 Features (Multi-line)")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_pdf = None
        self.current_blocks = []
        self.current_block_index = 0
        self.annotations = {}
        self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0, "Artifact": 0}
        
        # Label mappings
        self.label_map = {
            "1": "H1",
            "2": "H2", 
            "3": "H3",
            "4": "Title",
            "0": "Body",
            "9": "Artifact"  # New label for artifacts
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Select PDF", command=self.load_pdf).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1)
        
        # Progress info
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0/0")
        self.progress_label.grid(row=0, column=0)
        
        # Block info (NEW: shows multi-line information)
        self.block_info_label = ttk.Label(progress_frame, text="Block Info: Single line")
        self.block_info_label.grid(row=0, column=1, padx=(20, 0))
        
        # Artifact detection indicator
        self.artifact_label = ttk.Label(progress_frame, text="", foreground="red")
        self.artifact_label.grid(row=0, column=2, padx=(20, 0))
        
        # Annotation counts
        counts_frame = ttk.Frame(progress_frame)
        counts_frame.grid(row=0, column=3, padx=(20, 0))
        
        self.counts_label = ttk.Label(counts_frame, text="Title:0 H1:0 H2:0 H3:0 Body:0 Artifact:0")
        self.counts_label.grid(row=0, column=0)
        
        # Text display
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        ttk.Label(text_frame, text="Text Block (Multi-line Grouped):", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W)
        
        self.text_display = tk.Text(text_frame, height=15, width=80, wrap=tk.WORD, font=("Arial", 11))
        self.text_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_display.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.text_display.configure(yscrollcommand=scrollbar.set)
        
        # Annotation buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(button_frame, text="Annotate as:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=6, pady=(0, 5))
        
        # Create style for artifact button
        style = ttk.Style()
        
        # Try to configure the artifact button style
        try:
            style.configure("Artifact.TButton", foreground="red")
        except:
            # If style configuration fails, we'll use a regular button
            pass
        
        ttk.Button(button_frame, text="1 - H1", command=lambda: self.annotate("1"), width=10).grid(row=1, column=0, padx=2)
        ttk.Button(button_frame, text="2 - H2", command=lambda: self.annotate("2"), width=10).grid(row=1, column=1, padx=2)
        ttk.Button(button_frame, text="3 - H3", command=lambda: self.annotate("3"), width=10).grid(row=1, column=2, padx=2)
        ttk.Button(button_frame, text="4 - Title", command=lambda: self.annotate("4"), width=10).grid(row=1, column=3, padx=2)
        ttk.Button(button_frame, text="0 - Body", command=lambda: self.annotate("0"), width=10).grid(row=1, column=4, padx=2)
        
        # Use the styled button for artifact, with fallback
        try:
            ttk.Button(button_frame, text="9 - Artifact", command=lambda: self.annotate("9"), width=10, style="Artifact.TButton").grid(row=1, column=5, padx=2)
        except:
            # Fallback to regular button if styling fails
            ttk.Button(button_frame, text="9 - Artifact", command=lambda: self.annotate("9"), width=10).grid(row=1, column=5, padx=2)
        
        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Button(nav_frame, text="Previous", command=self.previous_block).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(nav_frame, text="Next", command=self.next_block).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(nav_frame, text="Save & Next PDF", command=self.save_and_next).grid(row=0, column=2)
        
        # Keyboard bindings
        self.root.bind('<Key-1>', lambda e: self.annotate("1"))
        self.root.bind('<Key-2>', lambda e: self.annotate("2"))
        self.root.bind('<Key-3>', lambda e: self.annotate("3"))
        self.root.bind('<Key-4>', lambda e: self.annotate("4"))
        self.root.bind('<Key-0>', lambda e: self.annotate("0"))
        self.root.bind('<Key-9>', lambda e: self.annotate("9"))  # New shortcut for artifacts
        self.root.focus_set()
        
    def load_pdf(self):
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if file_path:
            self.current_pdf = file_path
            self.file_label.config(text=f"File: {Path(file_path).name}")
            self.extract_grouped_blocks()
            
    def extract_grouped_blocks(self):
        """
        Extract grouped text blocks using the new multi-line grouping system.
        Now includes artifact detection.
        """
        try:
            # Import the updated feature extraction that groups lines
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from features import extract_document_adaptive_features, is_document_artifact
            
            # Use the updated feature extraction that groups lines
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
                    "text": features["text"],
                    "page": features["page_num"] + 1,
                    "block_id": i,
                    "line_count": features.get("line_count", 1),  # Multi-line info
                    "is_multiline": features.get("is_multiline", 0),
                    "is_likely_artifact": is_artifact  # New artifact indicator
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
    
    def display_current_block(self):
        if not self.current_blocks:
            return
            
        block = self.current_blocks[self.current_block_index]
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, f"Page {block['page']}\n\n{block['text']}")
        
        # Update progress
        total = len(self.current_blocks)
        current = self.current_block_index + 1
        self.progress_label.config(text=f"Progress: {current}/{total}")
        
        # Update block info (NEW: shows multi-line information)
        line_info = f"Lines: {block['line_count']}"
        if block['is_multiline']:
            line_info += " (MULTI-LINE BLOCK)"
        self.block_info_label.config(text=f"Block Info: {line_info}")
        
        # Update artifact indicator
        if block.get('is_likely_artifact', False):
            self.artifact_label.config(text="⚠️ Likely an artifact (header/footer/page number)")
        else:
            self.artifact_label.config(text="")
        
        # Update counts
        counts_text = " ".join([f"{k}:{v}" for k, v in self.annotation_counts.items()])
        self.counts_label.config(text=counts_text)
        
    def annotate(self, key):
        if not self.current_blocks:
            return
            
        label = self.label_map[key]
        block_id = self.current_blocks[self.current_block_index]["block_id"]
        
        # Remove old annotation if exists
        if block_id in self.annotations:
            old_label = self.annotations[block_id]
            self.annotation_counts[old_label] -= 1
            
        # Add new annotation
        self.annotations[block_id] = label
        self.annotation_counts[label] += 1
        
        # Auto-advance to next block
        self.next_block()
        
    def next_block(self):
        if self.current_block_index < len(self.current_blocks) - 1:
            self.current_block_index += 1
            self.display_current_block()
            
    def previous_block(self):
        if self.current_block_index > 0:
            self.current_block_index -= 1
            self.display_current_block()
            
    def save_and_next(self):
        if not self.current_pdf or not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return
            
        # Save annotations
        pdf_name = Path(self.current_pdf).stem
        output_file = f"data/annotated/{pdf_name}.json"
        
        # Create directory if not exists
        os.makedirs("data/annotated", exist_ok=True)
        
        # Prepare annotation data
        annotation_data = {
            "pdf_path": self.current_pdf,
            "total_blocks": len(self.current_blocks),
            "annotated_blocks": len(self.annotations),
            "multi_line_blocks": sum(1 for b in self.current_blocks if b['is_multiline']),
            "artifact_blocks": self.annotation_counts["Artifact"],
            "blocks": []
        }
        
        for i, block in enumerate(self.current_blocks):
            block_data = {
                "id": block["block_id"],
                "text": block["text"],
                "page": block["page"],
                "line_count": block["line_count"],
                "is_multiline": block["is_multiline"],
                "is_artifact": self.annotations.get(block["block_id"], "Body") == "Artifact",
                "role": self.annotations.get(block["block_id"], "Body")
            }
            annotation_data["blocks"].append(block_data)
            
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
        messagebox.showinfo(
            "Success", 
            f"Annotations saved to {output_file}\n"
            f"Multi-line blocks: {annotation_data['multi_line_blocks']}\n"
            f"Artifact blocks: {annotation_data['artifact_blocks']}"
        )
        
        # Reset for next PDF
        self.current_pdf = None
        self.current_blocks = []
        self.annotations = {}
        self.file_label.config(text="No file selected")
        self.text_display.delete(1.0, tk.END)
        self.progress_label.config(text="Progress: 0/0")
        self.block_info_label.config(text="Block Info: Single line")
        self.artifact_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFAnnotationTool(root)
    root.mainloop()