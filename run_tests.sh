#!/bin/bash
# Bash function to install pytest and run tests on the remote session

# Function to install pytest
install_pytest() {
    echo "Installing pytest..."
    pip install pytest
    echo "Pytest installed successfully!"
}

# Function to run all tests
run_tests() {
    echo "Running tests..."
    echo "=================="
    
    # Run data loader tests
    echo "Running data loader tests..."
    python -m pytest tests/test_data_loader.py -v
    
    echo ""
    echo "=================="
    
    # Run model tests
    echo "Running model tests..."
    python -m pytest tests/test_model.py -v
    
    echo ""
    echo "=================="
    echo "All tests completed!"
}

# Function to run specific test file
run_test_file() {
    if [ -z "$1" ]; then
        echo "Usage: run_test_file <test_file>"
        echo "Example: run_test_file tests/test_data_loader.py"
        return 1
    fi
    
    echo "Running test file: $1"
    python -m pytest "$1" -v
}

# Main execution
case "$1" in
    "install")
        install_pytest
        ;;
    "run")
        run_tests
        ;;
    "run-file")
        run_test_file "$2"
        ;;
    *)
        echo "Usage: $0 {install|run|run-file <test_file>}"
        echo ""
        echo "Examples:"
        echo "  $0 install          # Install pytest"
        echo "  $0 run              # Run all tests"
        echo "  $0 run-file tests/test_data_loader.py  # Run specific test file"
        ;;
esac
