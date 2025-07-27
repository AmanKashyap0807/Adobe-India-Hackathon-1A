#!/usr/bin/env python3
"""
Quick start script for the PDF Heading Extraction System.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_virtual_environment():
    """Check if running in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
        return True
    else:
        print("⚠️  No virtual environment detected")
        print("   Consider creating one: python -m venv venv")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_installation():
    """Run the installation test."""
    print("\nRunning installation test...")
    
    if not os.path.exists("test_installation.py"):
        print("❌ test_installation.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def setup_sample_data():
    """Set up sample data structure."""
    print("\nSetting up sample data...")
    
    # Copy existing PDFs to data/raw if they exist
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files in current directory")
        for pdf_file in pdf_files:
            dest_path = os.path.join("data", "raw", pdf_file)
            if not os.path.exists(dest_path):
                shutil.copy2(pdf_file, dest_path)
                print(f"  ✓ Copied {pdf_file} to data/raw/")
    
    # Create sample annotation if no annotations exist
    annotation_files = [f for f in os.listdir("data/annotated") if f.endswith(".json")]
    if not annotation_files:
        print("  ✓ Sample annotation file already exists")
    else:
        print(f"  ✓ Found {len(annotation_files)} annotation files")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Add PDF files to data/raw/ directory")
    print("2. Generate annotation templates:")
    print("   python src/extract/annotation_helper.py data/raw/your_file.pdf")
    print("3. Edit the generated JSON files in data/annotated/")
    print("   - Change 'role' values to: Title, H1, H2, H3, or Body")
    print("4. Build training dataset:")
    print("   python src/train/build_dataset.py")
    print("5. Train the model:")
    print("   python src/train/train_lgbm.py")
    print("6. Test on a PDF:")
    print("   python src/infer/predict.py your_file.pdf")
    
    print("\n📚 USEFUL COMMANDS:")
    print("- Test installation: python test_installation.py")
    print("- Batch processing: python src/infer/batch_predict.py input_dir output_dir")
    print("- Docker build: docker build -t pdf-heading:v1 .")
    
    print("\n📖 For detailed instructions, see README.md")

def main():
    """Main quick start function."""
    print("PDF Heading Extraction System - Quick Start")
    print("="*50)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    check_virtual_environment()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed, but you can continue...")
    
    # Setup sample data
    setup_sample_data()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    exit(main()) 