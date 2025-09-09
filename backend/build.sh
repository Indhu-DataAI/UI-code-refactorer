#!/bin/bash

# Set Cargo environment variables for Rust dependencies
export CARGO_HOME=/tmp/cargo
export CARGO_TARGET_DIR=/tmp/cargo-target

# Create directories
mkdir -p /tmp/cargo /tmp/cargo-target

# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
