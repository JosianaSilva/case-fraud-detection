#!/bin/bash

set -o errexit

echo "🔧 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🧪 Running tests..."
python -m pytest tests/ -v

echo "📊 Checking if models exist..."
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir -p models
fi

if [ ! -f "models/model.pkl" ] || [ ! -f "models/scaler.pkl" ]; then
    echo "🚀 No trained models found. Training new models..."
    python src/scripts/train.py
else
    echo "✅ Models found, skipping training..."
fi

echo "🏗️  Build completed successfully!"