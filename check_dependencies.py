#!/usr/bin/env python3
"""
Dependency check script for IRC BERT project.
Checks if required packages are installed in the current environment.
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"[OK] {package_name}: {version}")
            return True
        except ImportError:
            print(f"[FAIL] {package_name}: import failed")
            return False
    else:
        print(f"[FAIL] {package_name}: not installed")
        return False

def main():
    print("=== IRC BERT Dependency Check ===")
    print(f"Python version: {sys.version}")
    print()
    
    # Required packages
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
    ]
    
    results = []
    for package, import_name in packages:
        results.append(check_package(package, import_name))
    
    print()
    installed = sum(results)
    total = len(results)
    print(f"=== Summary: {installed}/{total} packages installed ===")
    
    if installed == total:
        print("[OK] All dependencies satisfied!")
        return 0
    else:
        print("[FAIL] Some dependencies missing.")
        print("Run setup.bat or setup.sh to install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())