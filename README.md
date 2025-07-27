# PDF Heading Extraction System

A machine learning-based system for extracting headings and generating outlines from PDF documents using LightGBM classification.

## Project Structure

```
├── data/
│   ├── raw/           # Unannotated PDFs
│   └── annotated/     # Same PDFs + labels.json per file
├── src/
│   ├── extract/       # Low-level text & feature extractors
│   ├── train/         # Dataset builder + LightGBM training script
│   └── infer/         # Fast runtime predictor
├── Dockerfile         # Slim base image (python:3.11-slim)
├── requirements.txt   # Dependencies
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify LightGBM installation
python -c "import lightgbm as lgb, numpy as np; print(lgb.LGBMClassifier(n_estimators=1).fit(np.array([[0]]),[0]))"
```

### 2. Dataset Creation (3 Hours)

#### 2.1 Collect PDFs
- Place 50-75 diverse PDFs in `data/raw/`
- Aim for different layouts: research papers, reports, brochures

#### 2.2 Quick Annotation
For each PDF, run:
```bash
python src/extract/annotation_helper.py data/raw/your_file.pdf
```

This creates `data/annotated/your_file.pdf.json` with all text blocks.

Edit the JSON file to change `"role"` values to:
- `"Title"` - Main document title
- `"H1"` - Level 1 headings
- `"H2"` - Level 2 headings  
- `"H3"` - Level 3 headings
- `"Body"` - Regular text (default)

### 3. Training

```bash
# Build dataset from annotations
python src/train/build_dataset.py

# Train LightGBM model
python src/train/train_lgbm.py
```

This creates `dataset.csv` and `model.txt`.

### 4. Inference

```bash
# Process a single PDF
python src/infer/predict.py path/to/your/document.pdf
```

Output: `document.json` with title and outline.

## Docker Usage

```bash
# Build image
docker build -t pdf-heading:v1 .

# Run inference
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-heading:v1
```

## Features

The system extracts 8 key features from each text block:

1. **font_ratio** - Font size relative to document median
2. **font_z** - Font size z-score within document
3. **word_count** - Number of words in block
4. **is_bold** - Whether text is bold (0/1)
5. **space_above_ratio** - Space above block relative to median font
6. **y_norm** - Vertical position normalized by page height
7. **starts_number** - Whether text starts with numbering (0/1)
8. **all_caps** - Whether text is all uppercase (0/1)

## Performance

- **Model size**: ~300 KB
- **Inference speed**: ~120ms for 20-page PDF
- **Docker image**: ~300 MB
- **Memory usage**: <200 MB

## File Formats

### Input
- PDF files in `data/raw/`

### Output
```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    {"level": "H3", "text": "Previous Work", "page": 2}
  ]
}
```

## Troubleshooting

1. **LightGBM OpenCL errors**: Use CPU-only build (default)
2. **Missing annotations**: Run annotation helper for each PDF
3. **Poor accuracy**: Add more diverse training data
4. **Memory issues**: Reduce batch size in training

## Dependencies

- pymupdf==1.23.8
- pdfplumber==0.10.3  
- numpy==1.24.3
- pandas==2.0.3
- lightgbm==4.1.0
- scikit-learn==1.3.0 