#!/bin/bash

# Paths (adjust if necessary)
VENV_DIR="venv"
API_DIR="api"
FRONTEND_DIR="frontend"

# Activate the virtual environment for the backend
if [ -d "$VENV_DIR" ]; then
  echo "Activating virtual environment..."
  source "$VENV_DIR/bin/activate"
else
  echo "Virtual environment not found! Please set it up in $VENV_DIR."
  exit 1
fi

# Run the backend in the background
echo "Starting backend (main.py)..."
python "$API_DIR/main.py" &
BACKEND_PID=$!

# Navigate to the frontend directory and start it with Node
echo "Starting frontend..."
cd "$FRONTEND_DIR" || exit
npm start &
FRONTEND_PID=$!

# Function to kill both processes on exit
cleanup() {
  echo "Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID
}
trap cleanup EXIT

# Wait for both processes
wait
