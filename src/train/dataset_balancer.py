import numpy as np
import pandas as pd
from collections import Counter
import json
import os

class DatasetBalancer:
    def __init__(self, target_distribution=None):
        # Target distribution based on research
        self.target_distribution = target_distribution or {
            "Body": 0.70,      # 70% - Most text blocks are paragraphs
            "H2": 0.12,        # 12% - Common subsection headings  
            "H1": 0.08,        # 8%  - Main section headings
            "H3": 0.07,        # 7%  - Sub-subsection headings
            "Title": 0.03      # 3%  - Usually 1 per document
        }
        self.min_samples_per_class = 500  # Minimum for LightGBM
        
    def check_balance(self, annotations):
        """Check current dataset balance"""
        if not annotations:
            return {}
            
        total = len(annotations)
        current_dist = {}
        
        for label in self.target_distribution.keys():
            count = sum(1 for ann in annotations if ann['role'] == label)
            current_dist[label] = count / total if total > 0 else 0
            
        return current_dist
    
    def needs_more_samples(self, current_counts, total_target=5000):
        """Determine which classes need more samples"""
        needs_more = {}
        
        for label, target_ratio in self.target_distribution.items():
            target_count = int(total_target * target_ratio)
            current_count = current_counts.get(label, 0)
            
            if current_count < max(target_count, self.min_samples_per_class):
                needs_more[label] = target_count - current_count
                
        return needs_more
    
    def get_balance_report(self, annotations):
        """Generate a comprehensive balance report"""
        if not annotations:
            return "No annotations found"
            
        current_dist = self.check_balance(annotations)
        current_counts = Counter(ann['role'] for ann in annotations)
        needs_more = self.needs_more_samples(current_counts)
        
        report = "Dataset Balance Report\n"
        report += "=" * 40 + "\n"
        report += f"Total samples: {len(annotations)}\n\n"
        
        report += "Current Distribution:\n"
        for label in self.target_distribution.keys():
            current_ratio = current_dist.get(label, 0)
            target_ratio = self.target_distribution[label]
            current_count = current_counts.get(label, 0)
            target_count = int(len(annotations) * target_ratio)
            
            status = "✓" if current_count >= target_count else "✗"
            report += f"  {label:6}: {current_count:4d} ({current_ratio:.1%}) "
            report += f"[Target: {target_ratio:.1%}] {status}\n"
        
        if needs_more:
            report += f"\nNeed more samples:\n"
            for label, count in needs_more.items():
                report += f"  {label}: +{count} samples\n"
        else:
            report += "\n✓ Dataset is well balanced!\n"
            
        return report
    
    def suggest_annotation_strategy(self, annotations):
        """Suggest which classes to focus on during annotation"""
        if not annotations:
            return "Start annotating any PDFs to build initial dataset"
            
        current_counts = Counter(ann['role'] for ann in annotations)
        needs_more = self.needs_more_samples(current_counts)
        
        if not needs_more:
            return "Dataset is well balanced. Continue with diverse PDFs."
        
        # Sort by priority (most needed first)
        sorted_needs = sorted(needs_more.items(), key=lambda x: x[1], reverse=True)
        
        strategy = "Annotation Strategy:\n"
        strategy += "Focus on these classes during annotation:\n\n"
        
        for label, count in sorted_needs[:3]:  # Top 3 priorities
            strategy += f"1. {label}: Need {count} more samples\n"
            
        strategy += "\nTips:\n"
        strategy += "- Look for documents with clear heading hierarchies\n"
        strategy += "- Prioritize documents with multiple heading levels\n"
        strategy += "- Include diverse document types (papers, reports, manuals)\n"
        
        return strategy
    
    def validate_annotations(self, annotation_file):
        """Validate annotation file format and content"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'blocks' not in data:
                return False, "Missing 'blocks' key"
            
            valid_roles = set(self.target_distribution.keys())
            invalid_roles = set()
            
            for block in data['blocks']:
                if 'role' not in block:
                    return False, f"Block missing 'role' field"
                if block['role'] not in valid_roles:
                    invalid_roles.add(block['role'])
            
            if invalid_roles:
                return False, f"Invalid roles found: {invalid_roles}"
            
            return True, "Valid annotation file"
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def get_annotation_stats(self, annotation_dir="data/annotated"):
        """Get statistics for all annotation files"""
        if not os.path.exists(annotation_dir):
            return "Annotation directory not found"
        
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        
        if not annotation_files:
            return "No annotation files found"
        
        total_blocks = 0
        role_counts = Counter()
        file_stats = []
        
        for file in annotation_files:
            file_path = os.path.join(annotation_dir, file)
            is_valid, message = self.validate_annotations(file_path)
            
            if is_valid:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_blocks = len(data['blocks'])
                total_blocks += file_blocks
                
                file_roles = Counter(block['role'] for block in data['blocks'])
                role_counts.update(file_roles)
                
                file_stats.append({
                    'file': file,
                    'blocks': file_blocks,
                    'roles': dict(file_roles)
                })
            else:
                file_stats.append({
                    'file': file,
                    'error': message
                })
        
        stats = {
            'total_files': len(annotation_files),
            'valid_files': len([f for f in file_stats if 'error' not in f]),
            'total_blocks': total_blocks,
            'role_distribution': dict(role_counts),
            'file_details': file_stats
        }
        
        return stats

def main():
    """Test the dataset balancer"""
    balancer = DatasetBalancer()
    
    # Example usage
    print("Dataset Balancer Test")
    print("=" * 30)
    
    # Test with sample data
    sample_annotations = [
        {'role': 'Body', 'text': 'Sample text 1'},
        {'role': 'Body', 'text': 'Sample text 2'},
        {'role': 'H1', 'text': 'Heading 1'},
        {'role': 'H2', 'text': 'Heading 2'},
        {'role': 'Title', 'text': 'Document Title'}
    ]
    
    print(balancer.get_balance_report(sample_annotations))
    print("\n" + balancer.suggest_annotation_strategy(sample_annotations))
    
    # Test annotation validation
    print("\nAnnotation Validation:")
    if os.path.exists("data/annotated/example.json"):
        is_valid, message = balancer.validate_annotations("data/annotated/example.json")
        print(f"example.json: {message}")

if __name__ == "__main__":
    main() 