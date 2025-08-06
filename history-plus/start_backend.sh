#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}        History Plus Backend Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Check if Python is installed
echo -e "${YELLOW}[1/4] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python is not installed or not in PATH${NC}"
    echo "Please install Python from https://python.org/downloads"
    echo "Or use your system package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"

# Navigate to backend directory
echo
echo -e "${YELLOW}[2/4] Navigating to backend directory...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

if [ ! -d "$(pwd)" ] || [ ! -f "app.py" ]; then
    echo -e "${RED}ERROR: Could not find backend directory or app.py${NC}"
    echo "Make sure this script is in the History Plus extension folder"
    exit 1
fi
echo -e "${GREEN}✓ Backend directory found${NC}"

# Install dependencies
echo
echo -e "${YELLOW}[3/4] Installing dependencies...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found${NC}"
    exit 1
fi

# Try pip3 first, then pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo -e "${RED}ERROR: pip is not installed${NC}"
    echo "Please install pip or use your system package manager"
    exit 1
fi

$PIP_CMD install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}WARNING: Some dependencies may have failed to install${NC}"
    echo "Attempting to continue anyway..."
fi

# Start the Flask server
echo
echo -e "${YELLOW}[4/4] Starting History Plus backend server...${NC}"
echo
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Backend starting on http://localhost:5000${NC}"
echo -e "${BLUE}  Press Ctrl+C to stop the server${NC}"
echo -e "${BLUE}================================================${NC}"
echo

$PYTHON_CMD main_app.py

echo
echo -e "${YELLOW}Backend stopped.${NC}"
read -p "Press Enter to exit..." 