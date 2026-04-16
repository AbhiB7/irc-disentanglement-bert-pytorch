#!/bin/bash
# Setup script for IRC Disentanglement PyTorch BERT project
# Installs dependencies and runs tests on the remote session

echo "=========================================="
echo "IRC Disentanglement - Setup and Test Script"
echo "=========================================="
echo ""

# Function to install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    echo "--------------------------"
    
    # Install pytest for testing
    echo "Installing pytest..."
    pip install pytest
    
    # Install transformers and related libraries (without version constraints)
    echo "Installing transformers and related libraries..."
    pip install transformers datasets sentence-transformers accelerate
    
    # Install additional utilities
    echo "Installing additional utilities..."
    pip install scikit-learn numpy pandas tqdm
    
    echo ""
    echo "Dependencies installed successfully!"
    echo ""
}

# Function to run all tests
run_tests() {
    echo "Running tests..."
    echo "----------------"
    
    # Run data loader tests
    echo "Running data loader tests..."
    python -m pytest tests/test_data_loader.py -v
    
    echo ""
    echo "----------------"
    
    # Run model tests
    echo "Running model tests..."
    python -m pytest tests/test_model.py -v
    
    echo ""
    echo "----------------"
    echo "All tests completed!"
}

# Function to run specific test file
run_test_file() {
    if [ -z "$1" ]; then
        echo "Usage: $0 run-file <test_file>"
        echo "Example: $0 run-file tests/test_data_loader.py"
        return 1
    fi
    
    echo "Running test file: $1"
    python -m pytest "$1" -v
}

# Main execution
case "$1" in
    "install")
        install_dependencies
        ;;
    "run")
        install_dependencies
        run_tests
        ;;
    "run-file")
        if [ -z "$2" ]; then
            echo "Usage: $0 run-file <test_file>"
            echo "Example: $0 run-file tests/test_data_loader.py"
            exit 1
        fi
        install_dependencies
        run_test_file "$2"
        ;;
    *)
        echo "Usage: $0 {install|run|run-file <test_file>}"
        echo ""
        echo "Options:"
        echo "  install          - Install all dependencies (pytest, transformers, etc.)"
        echo "  run              - Install dependencies and run all tests"
        echo "  run-file <file>  - Install dependencies and run specific test file"
        echo ""
        echo "Examples:"
        echo "  $0 install                    # Just install dependencies"
        echo "  $0 run                        # Install and run all tests"
        echo "  $0 run-file tests/test_data_loader.py  # Install and run specific test"
        ;;
esac
