#!/usr/bin/env python3
"""
Test script to verify the PDF heading extraction system installation.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import fitz
        print("✓ PyMuPDF (fitz) imported successfully")
    except ImportError as e:
        print(f"✗ PyMuPDF import failed: {e}")
        return False
    
    try:
        import pdfplumber
        print("✓ pdfplumber imported successfully")
    except ImportError as e:
        print(f"✗ pdfplumber import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print("✓ lightgbm imported successfully")
    except ImportError as e:
        print(f"✗ lightgbm import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    return True

def test_lightgbm():
    """Test LightGBM functionality."""
    print("\nTesting LightGBM...")
    
    try:
        import lightgbm as lgb
        import numpy as np
        
        # Test basic LightGBM functionality
        X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        y = np.array([0, 1, 0, 1])
        
        model = lgb.LGBMClassifier(n_estimators=1, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        pred = model.predict([[0, 1]])
        print(f"✓ LightGBM training and prediction successful: {pred}")
        
        # Test model saving
        model.booster_.save_model("test_model.txt")
        print("✓ LightGBM model saving successful")
        
        # Clean up
        import os
        if os.path.exists("test_model.txt"):
            os.remove("test_model.txt")
        
        return True
        
    except Exception as e:
        print(f"✗ LightGBM test failed: {e}")
        return False

def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    import os
    
    required_dirs = [
        "data/raw",
        "data/annotated", 
        "src/extract",
        "src/train",
        "src/infer"
    ]
    
    required_files = [
        "requirements.txt",
        "Dockerfile",
        "README.md",
        "src/extract/features.py",
        "src/extract/annotation_helper.py",
        "src/train/build_dataset.py",
        "src/train/train_lgbm.py",
        "src/infer/predict.py"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("PDF Heading Extraction System - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_lightgbm,
        test_project_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Place PDF files in data/raw/")
        print("2. Run: python src/extract/annotation_helper.py data/raw/your_file.pdf")
        print("3. Annotate the generated JSON files")
        print("4. Run: python src/train/build_dataset.py")
        print("5. Run: python src/train/train_lgbm.py")
        print("6. Test: python src/infer/predict.py your_file.pdf")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 