#!/bin/bash

# Pneumonia Detection System - Quick Setup Script
echo "🫁 Pneumonia Detection System - Quick Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# Create demo models for testing
echo "🧠 Creating demo models..."
python create_demo_model.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Download the chest X-ray dataset from Kaggle:"
echo "   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
echo ""
echo "2. Extract the dataset to the 'chest_xray' folder"
echo ""
echo "3. Train the models (optional, for better accuracy):"
echo "   python train_model.py"
echo ""
echo "4. Start the web application:"
echo "   python app.py"
echo ""
echo "5. Open your browser and go to: http://localhost:5000"
echo ""
echo "🩺 Happy pneumonia detection!"
