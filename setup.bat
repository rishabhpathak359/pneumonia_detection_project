@echo off
REM Pneumonia Detection System - Quick Setup Script for Windows

echo 🫁 Pneumonia Detection System - Quick Setup
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create models directory
echo 📁 Creating models directory...
if not exist models mkdir models

REM Create demo models for testing
echo 🧠 Creating demo models...
python create_demo_model.py

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Download the chest X-ray dataset from Kaggle:
echo    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
echo.
echo 2. Extract the dataset to the 'chest_xray' folder
echo.
echo 3. Train the models (optional, for better accuracy):
echo    python train_model.py
echo.
echo 4. Start the web application:
echo    python app.py
echo.
echo 5. Open your browser and go to: http://localhost:5000
echo.
echo 🩺 Happy pneumonia detection!
pause
