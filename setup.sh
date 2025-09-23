#!/bin/bash

# Setup script for Russian Speech Recognition with whisper.cpp
# This script installs system dependencies and sets up the environment

echo "Setting up Russian Speech Recognition with whisper.cpp..."
echo "=================================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux. Some features may not work on other platforms."
fi

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    git \
    make \
    build-essential \
    wget \
    curl

# Install Python dependencies (system packages)
echo "Installing Python dependencies..."
echo "Note: Using system packages instead of pip for better compatibility"

# Make main.py executable
chmod +x main.py

echo ""
echo "Setup completed successfully!"
echo ""
echo "To run the application:"
echo "  python3 main.py"
echo ""
echo "Note: The first run will download the model and compile whisper.cpp,"
echo "which may take a few minutes depending on your internet connection."