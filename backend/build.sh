#!/bin/bash

# Set Cargo environment variables for Rust dependencies
export CARGO_HOME=/tmp/cargo
export CARGO_TARGET_DIR=/tmp/cargo-target

# Create directories
mkdir -p /tmp/cargo /tmp/cargo-target

# Upgrade pip first
pip install --upgrade pip

echo "Installing Python packages with Cargo environment variables set..."

# Try to install with pre-compiled wheels first, fallback to source compilation
pip install --prefer-binary -r requirements.txt || {
    echo "Binary installation failed, trying with source compilation..."
    pip install --no-cache-dir -r requirements.txt
}
