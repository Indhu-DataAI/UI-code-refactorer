#!/bin/bash
# Build the frontend
echo "Building frontend..."
npm install
npm run build

# Install Python dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

echo "Build complete!"
