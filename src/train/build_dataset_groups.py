import pandas as pd
import glob
import json
import os
import numpy as np
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.features import extract_document_adaptive_features, FEATURE_COLUMNS

class GroupBasedDatasetBuilder:
    def __init__(self):
        self.total_groups_processed = 0
        self.total_lines_processed = 0
        self.efficiency_gains = []
        self.annotation_consistency = {}
        self.artifacts_filtered = 0
        
    def build_training_dataset_from_groups(self):
        """Build dataset from group-based annotations with enhanced efficiency tracking."""
        
        print("🔧 Building Training Dataset with Group-Based Annotations")
        print("=" * 60)
        
        all_features = []
        all_labels = []
        all_group_info = []
        
        # Process all annotated PDFs
        annotation_files = glob.glob("data/annotated/*.json")
        
        if not annotation_files:
            print("❌ No annotation files found in data/annotated/")
            return None
            
        print(f"📁 Found {len(annotation_files)} annotation files")
        
        for annotation_file in annotation_files:
            pdf_name = Path(annotation_file).stem
            pdf_file = f"data/raw/{pdf_name}.pdf"
            
            if not os.path.exists(pdf_file):
                print(f"⚠️  PDF file not found: {pdf_file}")
                continue
                
            print(f"\n📄 Processing {pdf_name}...")
            
            try:
                # Load group-based annotations
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Extract features using Two-Stage Hybrid Approach
                features_list, doc_stats = extract_document_adaptive_features(pdf_file)
                
                if not features_list:
                    print(f"⚠️  No features extracted from {pdf_name}")
                    continue
                
                # Process group-based annotations
                group_results = self.process_group_annotations(
                    features_list, annotations, pdf_name
                )
                
                if group_results:
                    all_features.extend(group_results['features'])
                    all_labels.extend(group_results['labels'])
                    all_group_info.extend(group_results['group_info'])
                    
                    # Track efficiency metrics
                    self.total_groups_processed += group_results['total_groups']
                    self.total_lines_processed += group_results['total_lines']
                    self.efficiency_gains.append(group_results['efficiency_gain'])
                    self.artifacts_filtered += group_results['artifacts_filtered']
                    
                    print(f"   ✅ Groups: {group_results['total_groups']} | Lines: {group_results['total_lines']} | Gain: {group_results['efficiency_gain']:.1f}x | Artifacts: {group_results['artifacts_filtered']}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_name}: {str(e)}")
                continue
        
        if not all_features:
            print("❌ No features extracted from any PDF")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        df['label'] = all_labels
        
        # Add group information columns
        group_df = pd.DataFrame(all_group_info)
        df = pd.concat([df, group_df], axis=1)
        
        # Save dataset
        output_file = "data/training_dataset_groups.csv"
        df.to_csv(output_file, index=False)
        
        # Generate comprehensive report
        self.generate_efficiency_report(df, output_file)
        
        print(f"\n✅ Dataset created successfully!")
        print(f"📊 Total samples: {len(df)}")
        print(f"📊 Total groups processed: {self.total_groups_processed}")
        print(f"📊 Total lines processed: {self.total_lines_processed}")
        print(f"📊 Artifacts filtered: {self.artifacts_filtered}")
        print(f"📊 Average efficiency gain: {np.mean(self.efficiency_gains):.1f}x")
        
        return df
    
    def process_group_annotations(self, features_list, annotations, pdf_name):
        """Process group-based annotations and create individual training samples."""
        
        # Check if this is a group-based annotation or legacy format
        if 'groups' in annotations:
            return self.process_new_group_format(features_list, annotations, pdf_name)
        else:
            return self.process_legacy_format(features_list, annotations, pdf_name)
    
    def process_new_group_format(self, features_list, annotations, pdf_name):
        """Process new group-based annotation format."""
        
        features = []
        labels = []
        group_info = []
        artifacts_filtered = 0
        
        # Create mapping from group_id to role
        group_role_map = {}
        is_artifact_map = {}
        for group in annotations.get('groups', []):
            group_role_map[group['group_id']] = group['role']
            is_artifact_map[group['group_id']] = group.get('is_artifact', False) or group['role'] == 'Artifact'
        
        # Process each feature (which represents a group)
        for i, feature_dict in enumerate(features_list):
            # Get the role and artifact status for this group
            role = group_role_map.get(i, 'Body')
            is_artifact = is_artifact_map.get(i, False)
            
            # Skip artifacts
            if is_artifact:
                artifacts_filtered += 1
                continue
            
            group_info.append({
                'pdf_name': pdf_name,
                'group_id': i,
                'line_count': feature_dict.get('line_count', 1),
                'is_multiline': feature_dict.get('is_multiline', 0),
                'line_font_consistency': feature_dict.get('line_font_consistency', 1.0),
                'annotation_role': role
            })
        
        return {
            'features': features,
            'labels': labels,
            'group_info': group_info,
            'total_groups': len(features_list),
            'total_lines': sum(f.get('line_count', 1) for f in features_list),
            'efficiency_gain': annotations.get('efficiency_gain', 1.0)
        }
    
    def process_legacy_format(self, features_list, annotations, pdf_name):
        """Process legacy annotation format for backward compatibility."""
        
        features = []
        labels = []
        group_info = []
        
        # Create label mapping from legacy format
        label_map = {block['id']: block['role'] for block in annotations.get('blocks', [])}
        
        # Process each feature
        for i, feature_dict in enumerate(features_list):
            # Get the role for this block
            role = label_map.get(i, 'Body')
            
            # Create feature vector
            feature_vector = {}
            for col in FEATURE_COLUMNS:
                if col in feature_dict:
                    feature_vector[col] = feature_dict[col]
                else:
                    feature_vector[col] = 0.0
            
            features.append(feature_vector)
            labels.append(role)
            
            # Add group information (legacy format assumed single-line)
            group_info.append({
                'pdf_name': pdf_name,
                'group_id': i,
                'line_count': 1,
                'is_multiline': 0,
                'line_font_consistency': 1.0,
                'annotation_role': role
            })
        
        return {
            'features': features,
            'labels': labels,
            'group_info': group_info,
            'total_groups': len(features_list),
            'total_lines': len(features_list),
            'efficiency_gain': 1.0  # Legacy format has no efficiency gain
        }
    
    def generate_efficiency_report(self, df, output_file):
        """Generate comprehensive efficiency and quality report."""
        
        report = {
            "dataset_info": {
                "total_samples": len(df),
                "total_groups_processed": self.total_groups_processed,
                "total_lines_processed": self.total_lines_processed,
                "average_efficiency_gain": np.mean(self.efficiency_gains),
                "output_file": output_file
            },
            "feature_validation": {
                "expected_features": len(FEATURE_COLUMNS),
                "actual_features": len([col for col in FEATURE_COLUMNS if col in df.columns]),
                "feature_columns": FEATURE_COLUMNS
            },
            "label_distribution": df['label'].value_counts().to_dict(),
            "multi_line_statistics": {
                "total_multi_line_groups": len(df[df['is_multiline'] == 1]),
                "average_line_count": df['line_count'].mean(),
                "font_consistency_stats": {
                    "mean": df['line_font_consistency'].mean(),
                    "std": df['line_font_consistency'].std(),
                    "min": df['line_font_consistency'].min(),
                    "max": df['line_font_consistency'].max()
                }
            },
            "efficiency_analysis": {
                "efficiency_gains": self.efficiency_gains,
                "time_savings": {
                    "estimated_annotation_time_without_groups": f"{self.total_lines_processed * 0.1:.1f} minutes",
                    "estimated_annotation_time_with_groups": f"{self.total_groups_processed * 0.1:.1f} minutes",
                    "time_saved": f"{(self.total_lines_processed - self.total_groups_processed) * 0.1:.1f} minutes"
                }
            }
        }
        
        # Save report
        report_file = "data/dataset_groups_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Efficiency Report saved to: {report_file}")
        print(f"⏱️  Estimated time saved: {report['efficiency_analysis']['time_savings']['time_saved']}")
        print(f"📈 Average efficiency gain: {report['dataset_info']['average_efficiency_gain']:.1f}x")
        
        return report

def main():
    """Main function to build the group-based dataset."""
    
    print("🚀 Group-Based Dataset Builder")
    print("=" * 40)
    
    # Check if required directories exist
    if not os.path.exists("data/raw"):
        print("❌ data/raw directory not found")
        return
    
    if not os.path.exists("data/annotated"):
        print("❌ data/annotated directory not found")
        return
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Build dataset
    builder = GroupBasedDatasetBuilder()
    dataset = builder.build_training_dataset_from_groups()
    
    if dataset is not None:
        print(f"\n✅ Dataset successfully created with {len(dataset)} samples")
        print(f"📁 Saved to: data/training_dataset_groups.csv")
        print(f"📊 Label distribution:")
        print(dataset['label'].value_counts())
        
        # Validate feature consistency
        missing_features = [col for col in FEATURE_COLUMNS if col not in dataset.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        else:
            print(f"✅ All {len(FEATURE_COLUMNS)} features present")
    else:
        print("❌ Failed to create dataset")

if __name__ == "__main__":
    main() 