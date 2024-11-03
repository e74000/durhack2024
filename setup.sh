#!/bin/bash

# Define paths
VENV_DIR="venv"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment already exists. Skipping creation."
else
  # Create the virtual environment
  echo "Creating virtual environment..."
  python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install pandas numpy torch flask scikit-learn flask-cors

echo "Setup complete! Your virtual environment is ready."

# Reminder for activating the virtual environment
echo "To activate the virtual environment, run:"
echo "source $VENV_DIR/bin/activate"
