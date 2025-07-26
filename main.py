from core.parser import extract_text_blocks
from core.rules import apply_rules
from core.features import create_features
import json
import os

def run_extraction(pdf_path: str, output_dir: str):
    """
    Runs the full PDF structure extraction pipeline.
    Step 1: Parse raw text blocks.
    Step 2: Apply rules to tag blocks.
    Step 3: Generate features for the ML model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Step 1: Foundational PDF Parsing ---
    print(f"Starting extraction from '{pdf_path}'...")
    blocks = extract_text_blocks(pdf_path)
    print(f"Extracted {len(blocks)} text blocks.")
    step1_output_path = os.path.join(output_dir, "step1_extracted_blocks.json")
    with open(step1_output_path, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved raw blocks to '{step1_output_path}'")

    # --- Step 2: Rule-Based Structuring ---
    print("\nApplying rule-based structuring...")
    enhanced_blocks = apply_rules(blocks)
    print(f"Assigned tags to {len(enhanced_blocks)} blocks.")
    step2_output_path = os.path.join(output_dir, "step2_rule_based_tags.json")
    # We need to convert our Pydantic objects back to dicts for JSON serialization
    enhanced_blocks_dict = [block.model_dump() for block in enhanced_blocks]
    with open(step2_output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_blocks_dict, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved rule-based tags to '{step2_output_path}'")

    # --- Step 3: Feature Engineering ---
    print("\nGenerating features for machine learning...")
    features_df = create_features(enhanced_blocks)
    print(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} blocks.")
    step3_output_path = os.path.join(output_dir, "step3_features.csv")
    features_df.to_csv(step3_output_path, index=False, encoding='utf-8')
    print(f"Successfully saved features to '{step3_output_path}'")

    print("\n--- Next Steps ---")
    print("1. Manually create the labeled dataset by running:")
    print(f"   python scripts/create_dataset.py")
    print("2. This will create 'dataset/training_data.csv' which is needed for Step 4.")

if __name__ == "__main__":
    run_extraction("data/sample.pdf", "output")
