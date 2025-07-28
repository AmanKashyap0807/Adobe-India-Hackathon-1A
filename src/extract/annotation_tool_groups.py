import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.features import extract_document_adaptive_features

class GroupBasedAnnotationTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF Text Block Annotation Tool - Group-Based (20 Features)")
        self.root.geometry("1000x800")
        
        # Data storage
        self.current_pdf = None
        self.current_groups = []  # Grouped blocks for annotation
        self.current_group_index = 0
        self.annotations = {}  # group_id -> label
        self.annotation_counts = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "Body": 0}
        
        # Label mapping
        self.label_map = {
            "1": "H1",
            "2": "H2", 
            "3": "H3",
            "4": "Title",
            "0": "Body"
        }
        
        self.setup_ui()
        self.setup_keyboard_shortcuts()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Load PDF", command=self.load_pdf).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky=tk.W)
        
        # Efficiency statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Efficiency Statistics", padding="5")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.efficiency_label = ttk.Label(stats_frame, text="Total Groups: 0 | Total Lines: 0 | Speed Gain: 0x")
        self.efficiency_label.grid(row=0, column=0, sticky=tk.W)
        
        # Text display area
        text_frame = ttk.LabelFrame(main_frame, text="Current Text Group", padding="5")
        text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.text_display = tk.Text(text_frame, height=8, width=80, wrap=tk.WORD)
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for text
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_display.yview)
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        
        # Group information
        group_frame = ttk.LabelFrame(main_frame, text="Group Information", padding="5")
        group_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.group_info_label = ttk.Label(group_frame, text="Group: 0/0 | Lines: 0 | Type: None")
        self.group_info_label.grid(row=0, column=0, sticky=tk.W)
        
        # Individual line preview
        lines_frame = ttk.LabelFrame(main_frame, text="Individual Lines Preview", padding="5")
        lines_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.lines_preview = tk.Text(lines_frame, height=4, width=80, wrap=tk.WORD)
        self.lines_preview.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress and controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(control_frame, text="Progress: 0/0")
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="Previous", command=self.previous_group).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_group).grid(row=0, column=2, padx=5)
        
        # Annotation counts
        counts_frame = ttk.LabelFrame(main_frame, text="Annotation Counts", padding="5")
        counts_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.counts_label = ttk.Label(counts_frame, text="Title: 0 | H1: 0 | H2: 0 | H3: 0 | Body: 0")
        self.counts_label.grid(row=0, column=0, sticky=tk.W)
        
        # Save button
        save_frame = ttk.Frame(main_frame)
        save_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(save_frame, text="Save Annotations", command=self.save_annotations).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(save_frame, text="Exit", command=self.root.quit).grid(row=0, column=1)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        lines_frame.columnconfigure(0, weight=1)
        lines_frame.rowconfigure(0, weight=1)
        
    def setup_keyboard_shortcuts(self):
        self.root.bind('<Key-1>', lambda e: self.annotate("1"))
        self.root.bind('<Key-2>', lambda e: self.annotate("2"))
        self.root.bind('<Key-3>', lambda e: self.annotate("3"))
        self.root.bind('<Key-4>', lambda e: self.annotate("4"))
        self.root.bind('<Key-0>', lambda e: self.annotate("0"))
        self.root.bind('<Left>', lambda e: self.previous_group())
        self.root.bind('<Right>', lambda e: self.next_group())
        self.root.bind('<space>', lambda e: self.next_group())
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
        """Extract grouped text blocks using the Two-Stage Hybrid Approach."""
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
                text=f"Total Groups: {len(self.current_groups)} | Total Lines: {total_lines} | Speed Gain: {speed_gain:.1f}x"
            )
            
            self.display_current_group()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
            
    def display_current_group(self):
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
        self.group_info_label.config(
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
        
    def annotate(self, key):
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
        self.next_group()
        
    def next_group(self):
        if self.current_groups and self.current_group_index < len(self.current_groups) - 1:
            self.current_group_index += 1
            self.display_current_group()
            
    def previous_group(self):
        if self.current_groups and self.current_group_index > 0:
            self.current_group_index -= 1
            self.display_current_group()
            
    def save_annotations(self):
        if not self.current_pdf or not self.current_groups:
            messagebox.showwarning("Warning", "No PDF loaded or no groups to save")
            return
            
        # Create output directory
        output_dir = "data/annotated"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        pdf_name = Path(self.current_pdf).stem
        output_file = os.path.join(output_dir, f"{pdf_name}.json")
        
        # Prepare annotation data
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
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Success", f"Annotations saved to {output_file}\n"
                               f"Groups annotated: {len(self.annotations)}/{len(self.current_groups)}\n"
                               f"Efficiency gain: {annotation_data['efficiency_gain']:.1f}x")
                               
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GroupBasedAnnotationTool()
    app.run() 