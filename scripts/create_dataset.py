import json
import pandas as pd
import os

# Define the mapping from numbers to semantic tags
TAG_MAP = {
    '1': 'paragraph',
    '2': 'h1',
    '3': 'h2',
    '4': 'h3',
    '5': 'list_item',
    '6': 'footer',
    '7': 'header',
    '8': 'caption',
    '9': 'table',
    '10': 'code',
    '11': 'title',
    '0': 'other', # For elements that don't fit other categories
}

def create_labeled_dataset(input_json_path: str, output_csv_path: str):
    """
    A command-line tool to manually label text blocks for creating a training dataset.

    It loads blocks from a JSON file, displays each one to the user, and prompts
    for the correct tag. The results are saved to a CSV file.

    Args:
        input_json_path: Path to the JSON file with rule-based tags.
        output_csv_path: Path to save the final labeled CSV data.
    """
    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found at '{input_json_path}'")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    print("--- Starting Manual Labeling ---\n")
    print("Please use the following numbers to label the blocks:")
    for key, value in TAG_MAP.items():
        print(f"  {key}: {value}")
    print("\nFor each block, enter the number for the correct tag or press Enter to accept the suggestion.")

    labeled_data = []
    for block in blocks:
        print("\n" + "-" * 60)
        print(f"Block ID: {block['block_id']} | Page: {block['page_num']}")
        print(f"Text: \n{block['text']}")
        print("-" * 60)
        
        suggested_tag = block.get('tag', 'paragraph')
        
        final_tag = None
        while final_tag is None:
            user_input = input(f"Suggested: '{suggested_tag}'. Enter number (or Enter to accept): ")
            
            if not user_input.strip():
                final_tag = suggested_tag
                break
            elif user_input.strip() in TAG_MAP:
                final_tag = TAG_MAP[user_input.strip()]
            else:
                print(f"  [Invalid input] Please enter a number from the list above.")
        
        block['corrected_tag'] = final_tag
        labeled_data.append(block)
        print(f"==> Tagged as: '{final_tag}'\n")

    # Save to CSV
    df = pd.DataFrame(labeled_data)
    # Select relevant columns for the final dataset
    final_df = df[['block_id', 'text', 'corrected_tag']]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"--- Labeling Complete ---")
    print(f"Labeled dataset saved to '{output_csv_path}'")

if __name__ == "__main__":
    input_file = "output/step2_rule_based_tags.json"
    output_file = "dataset/training_data.csv"
    create_labeled_dataset(input_file, output_file)
