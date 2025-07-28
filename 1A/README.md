# Adobe India Hackathon 2025 - Challenge 1A: PDF Structure Extraction

## Overview

This solution implements an intelligent PDF structure extraction system that automatically identifies and extracts document titles and hierarchical headings (H1, H2, H3) from PDF documents. The system uses document-adaptive machine learning to understand each PDF's unique formatting patterns and produce structured JSON output.

## Approach

### Core Methodology
- **Document-Adaptive Feature Engineering**: Each PDF is analyzed to understand its unique font distributions, spacing patterns, and layout characteristics
- **22-Dimensional Feature Vector**: Comprehensive feature extraction including font properties, positioning, text content analysis, and semantic grouping
- **LightGBM Classification**: Fast, efficient gradient boosting model trained on diverse document types
- **Hierarchical Consistency Rules**: Post-processing ensures logical heading structure (H1 → H2 → H3)

### Technical Architecture
1. **PDF Text Extraction**: Uses PyMuPDF for robust text and layout extraction
2. **Feature Engineering**: 
   - Font statistics (size, weight, style)
   - Positional features (normalized coordinates, alignment)
   - Text content analysis (length, word count, case patterns)
   - Semantic grouping (bullet lists, tables, multi-line blocks)
3. **Machine Learning Pipeline**: LightGBM classifier with document-specific normalization
4. **Output Generation**: Structured JSON with title and hierarchical outline

## Models and Libraries Used

### Core Libraries
- **PyMuPDF (fitz)**: PDF text extraction and layout analysis
- **LightGBM**: Gradient boosting classifier for heading detection
- **NumPy/Pandas**: Data processing and feature engineering
- **Scikit-learn**: Model evaluation and preprocessing utilities

### Model Details
- **Type**: LightGBM Gradient Boosting Classifier
- **Size**: <200MB (optimized for Docker constraints)
- **Features**: 22-dimensional feature vector per text block
- **Classes**: 5 (Body, Title, H1, H2, H3)
- **Training Data**: 50+ diverse PDF documents with manual annotations

### Feature Categories
1. **Font & Style** (6 features): Size statistics, bold/italic detection
2. **Position** (4 features): Normalized coordinates, spacing ratios
3. **Text Content** (4 features): Length, word count, numbering patterns
4. **Context** (3 features): Text density, whitespace analysis
5. **Semantic Group** (5 features): Block type, element counts, aspect ratios

## Performance Characteristics

- **Processing Time**: <10 seconds for 50-page PDFs
- **Model Size**: <200MB (meets challenge constraints)
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU Architecture**: AMD64 compatible
- **Network**: Fully offline operation

## Build and Run Instructions

### Prerequisites
- Docker with AMD64 support
- 8GB+ available RAM
- 2GB+ available disk space

### Build Command
```bash
docker build --platform linux/amd64 -t pdf-structure-extractor:latest .
```

### Run Command
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-structure-extractor:latest
```

### Input/Output Structure
- **Input**: Place PDF files in `./input/` directory
- **Output**: JSON files generated in `./output/` directory
- **Format**: `filename.pdf` → `filename.json`

### Expected Output Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Sub Heading",
      "page": 2
    },
    {
      "level": "H3",
      "text": "Sub-sub Heading", 
      "page": 3
    }
  ]
}
```

## Testing

### Local Testing
```bash
# Create test directories
mkdir -p input output

# Copy test PDFs to input directory
cp your_test_files.pdf input/

# Build and run
docker build --platform linux/amd64 -t pdf-extractor .
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-extractor

# Check results
ls output/
cat output/*.json
```

### Validation Checklist
- [x] All PDFs in input directory processed
- [x] JSON output files generated for each PDF
- [x] Output format matches required structure
- [x] Processing completes within 10 seconds for 50-page PDFs
- [x] Solution works without internet access
- [x] Memory usage stays within 16GB limit
- [x] Compatible with AMD64 architecture

## Technical Implementation Details

### Feature Extraction Pipeline
1. **PDF Parsing**: Extract text blocks with font and position metadata
2. **Artifact Filtering**: Remove headers, footers, page numbers
3. **Block Grouping**: Merge related text spans into semantic blocks
4. **Feature Computation**: Calculate 22 features per block
5. **Document Normalization**: Z-score normalization using document statistics

### Model Training
- **Dataset**: 50+ diverse PDF documents
- **Annotation Tool**: Custom web-based labeling interface
- **Cross-validation**: 5-fold CV for model selection
- **Hyperparameter Tuning**: Bayesian optimization for LightGBM parameters

### Performance Optimizations
- **Efficient PDF Processing**: Stream-based text extraction
- **Memory Management**: Lazy loading and garbage collection
- **Parallel Processing**: Multi-threaded feature extraction
- **Model Optimization**: Quantized LightGBM model for size reduction

## License
This project is developed for the Adobe India Hackathon 2025 and uses open-source libraries as specified in requirements.txt.
