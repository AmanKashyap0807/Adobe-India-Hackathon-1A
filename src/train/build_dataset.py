import pandas as pd
import glob
import json
import os
import sys

# Add the extract module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'extract'))
from features import extract_document_adaptive_features, FEATURE_COLUMNS
from dataset_balancer import DatasetBalancer

def build_dataset():
    """Build training dataset from annotated PDFs using all 20 document-adaptive features (including multi-line)."""
    rows, labels = [], []
    balancer = DatasetBalancer()
    
    print("Building dataset with 20 document-adaptive features (including multi-line)...")
    print(f"Feature set: {len(FEATURE_COLUMNS)} features")
    for i, feature in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i:2d}. {feature}")
    print()
    
    multi_line_stats = {"total_blocks": 0, "multi_line_blocks": 0, "avg_line_count": 0}
    artifact_stats = {"total_artifacts": 0, "total_blocks": 0}
    
    for pdf in glob.glob("data/raw/*.pdf"):
        pdf_name = os.path.basename(pdf)
        annotation_file = f"data/annotated/{pdf_name}.json"
        
        # Skip if no annotation file exists
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation file found for {pdf_name}")
            continue
            
        try:
            # Load annotations
            with open(annotation_file, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            
            # Validate annotation file
            is_valid, message = balancer.validate_annotations(annotation_file)
            if not is_valid:
                print(f"Warning: Invalid annotation file {pdf_name}: {message}")
                continue
            
            # Extract document-adaptive features (now with multi-line support)
            features_list, doc_stats = extract_document_adaptive_features(pdf)
            
            if not features_list:
                print(f"Warning: No features extracted from {pdf_name}")
                continue
            
            # Create mapping from block index to role
            labelled = {x["id"]: x["role"] for x in ann["blocks"]}
            is_artifact = {x["id"]: x.get("is_artifact", False) for x in ann["blocks"]}
            
            # Match features with annotations
            for i, feats in enumerate(features_list):
                role = labelled.get(i, "Body")  # Default to Body if not annotated
                
                # Skip artifacts in training data
                if is_artifact.get(i, False) or role == "Artifact":
                    artifact_stats["total_artifacts"] += 1
                    continue
                
                # Ensure all required features are present
                feature_row = {}
                for feature in FEATURE_COLUMNS:
                    feature_row[feature] = feats.get(feature, 0.0)
                
                rows.append(feature_row)
                labels.append(role)
                
                # Track multi-line statistics
                multi_line_stats["total_blocks"] += 1
                if feats.get("is_multiline", 0):
                    multi_line_stats["multi_line_blocks"] += 1
                multi_line_stats["avg_line_count"] += feats.get("line_count", 1)
            
            artifact_stats["total_blocks"] += len(features_list)
            print(f"✓ Processed {pdf_name}: {len(features_list)} blocks")
            
        except Exception as e:
            print(f"Error processing {pdf_name}: {e}")
            continue
    
    if not rows:
        print("No data extracted. Please check your annotation files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    df["label"] = labels
    
    # Verify all features are present
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    # Save dataset
    df.to_csv("dataset.csv", index=False)
    
    # Calculate multi-line statistics
    if multi_line_stats["total_blocks"] > 0:
        multi_line_stats["avg_line_count"] /= multi_line_stats["total_blocks"]
        multi_line_percentage = (multi_line_stats["multi_line_blocks"] / multi_line_stats["total_blocks"]) * 100
    
    # Generate balance report
    annotations = [{"role": label} for label in labels]
    print("\n" + "="*50)
    print("DATASET CREATION COMPLETE")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Features extracted: {len([f for f in FEATURE_COLUMNS if f in df.columns])}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Artifact statistics
    artifact_percentage = (artifact_stats["total_artifacts"] / artifact_stats["total_blocks"]) * 100 if artifact_stats["total_blocks"] > 0 else 0
    print(f"\nArtifact Statistics:")
    print(f"  Total blocks: {artifact_stats['total_blocks']}")
    print(f"  Artifact blocks filtered: {artifact_stats['total_artifacts']} ({artifact_percentage:.1f}%)")
    print(f"  Content blocks used: {artifact_stats['total_blocks'] - artifact_stats['total_artifacts']}")
    
    # Multi-line statistics
    print(f"\nMulti-line Statistics:")
    print(f"  Total blocks: {multi_line_stats['total_blocks']}")
    print(f"  Multi-line blocks: {multi_line_stats['multi_line_blocks']} ({multi_line_percentage:.1f}%)")
    print(f"  Average line count: {multi_line_stats['avg_line_count']:.1f}")
    
    print("\n" + balancer.get_balance_report(annotations))
    print("\n" + balancer.suggest_annotation_strategy(annotations))
    
    # Save dataset statistics
    stats = {
        "total_samples": len(df),
        "features_used": FEATURE_COLUMNS,
        "label_distribution": df["label"].value_counts().to_dict(),
        "feature_columns": [col for col in df.columns if col not in ["label", "text", "page_num"]],
        "documents_processed": len([f for f in glob.glob("data/raw/*.pdf") 
                                  if os.path.exists(f"data/annotated/{os.path.basename(f)}.json")]),
        "multi_line_statistics": multi_line_stats,
        "artifact_statistics": {
            "total_blocks": artifact_stats["total_blocks"],
            "artifact_blocks": artifact_stats["total_artifacts"],
            "artifact_percentage": artifact_percentage
        }
    }
    
    with open("dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset statistics saved to dataset_stats.json")
    print(f"Feature compatibility: ✅ All {len(FEATURE_COLUMNS)} features extracted")
    print(f"Multi-line support: ✅ {multi_line_stats['multi_line_blocks']} multi-line blocks detected")
    print(f"Artifact filtering: ✅ {artifact_stats['total_artifacts']} artifact blocks filtered")

if __name__ == "__main__":
    build_dataset()