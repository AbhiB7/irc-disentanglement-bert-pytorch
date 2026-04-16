# IRC BERT Project - Setup & Instructions

## Quick Start

### 1. Activate Environment

#### Windows
```bat
call .venv-irc-bert\Scripts\activate.bat
```

#### Linux/Mac
```bash
source .venv-irc-bert/bin/activate
```

### 2. Run Tests
```bash
python tests/test_data_loader.py
python tests/test_model.py
```

### 3. Train Model
```bash
# Dev-only dry run
python src/train.py --dev-only data/dev/2004-11-15_03 --test-end 100

# Minimal test (10 pairs only)
python src/train.py --mode dev-only --data-dir data --test-end 10

# Full training (on GPU/Vast.ai)
python src/train.py --mode train --data-dir data
```

---

## Windows Setup (setup.bat)

```bat
@echo off
REM Setup script for IRC Disentanglement PyTorch BERT project on Windows
REM Updated for Python 3.13 compatibility (2026-04-15)

if not exist ".venv-irc-bert" (
    python -m venv .venv-irc-bert
)
call .venv-irc-bert\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch (CPU version for compatibility with Python 3.13)
pip install torch torchvision torchaudio

REM Core ML dependencies (updated for Python 3.13 compatibility)
REM transformers 5.5.4, sentence-transformers 5.4.1 support Python 3.13
pip install transformers==5.5.4 datasets sentence-transformers==5.4.1 accelerate

REM Additional utils
pip install scikit-learn numpy pandas tqdm

echo Setup complete. Run 'call .venv-irc-bert\Scripts\activate.bat' to activate.
echo To train: python train.py --help
```

**Save as `setup.bat` and run.**

---

## Linux/Remote Setup (setup.sh)

```bash
#!/bin/bash
# Setup script for IRC Disentanglement PyTorch BERT project on Linux/Remote (e.g., Vast.ai)
# Updated for Python 3.13 compatibility (2026-04-15)

set -e

if [ ! -d ".venv-irc-bert" ]; then
    python3 -m venv .venv-irc-bert
fi

source .venv-irc-bert/bin/activate

python -m pip install --upgrade pip setuptools wheel

# PyTorch (CPU version for compatibility with Python 3.13)
pip install torch torchvision torchaudio

# Core ML deps (updated for Python 3.13 compatibility)
# transformers 5.5.4, sentence-transformers 5.4.1 support Python 3.13
pip install transformers==5.5.4 datasets sentence-transformers==5.4.1 accelerate

# Utils
pip install scikit-learn numpy pandas tqdm

echo "Setup complete. Source .venv-irc-bert/bin/activate to activate."
echo "To train: python train.py --help"
```

**Save as `setup.sh`, `chmod +x setup.sh`, run.**

---

## Local CPU Setup (No GPU)

For local laptop testing (CPU-only). Slow training, but dev dry-runs feasible (~5-10min per epoch on 254-msg dev file).

### Windows (local_setup.bat)

```bat
@echo off
REM Local CPU setup for IRC BERT (no GPU needed)
REM Updated for Python 3.13 compatibility (2026-04-15)

if not exist ".venv-irc-bert" (
    python -m venv .venv-irc-bert
)
call .venv-irc-bert\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

REM CPU-only PyTorch (stable, fast install)
pip install torch torchvision torchaudio

REM HF libs (CPU compatible, updated for Python 3.13)
pip install transformers==5.5.4 datasets sentence-transformers==5.4.1 accelerate

REM Utils
pip install scikit-learn numpy pandas tqdm

echo Local CPU setup complete.
echo Activate: call .venv-irc-bert\Scripts\activate.bat
echo Test: python -c "import torch; print('CPU:', torch.cuda.is_available() == False)"
```

### Linux/Remote GPU (vast_setup.sh - later)
Use [setup.sh](#linuxremote-setup-setupsh) CUDA version for Vast.ai RTX 4090.

---

## Verify Installation

### Windows
```bat
.venv-irc-bert\Scripts\activate.bat
```

### Linux/Mac
```bash
source .venv-irc-bert/bin/activate
```

### Test Commands
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from sentence_transformers import CrossEncoder; print('OK')"
python -c "from data_loader import IRCDisentanglementDataset; print('Data loader OK')"
```

### Verify CPU
```bash
python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('bert-base-uncased', num_labels=1); print('CPU BERT OK')"
python -c "from data_loader import IRCDisentanglementDataset; print('Data loader OK')"
```

---

## Running Tests

```bash
python tests/test_data_loader.py
python tests/test_model.py
```

### Smoke Test Results (2026-04-16)

✅ **Smoke test completed successfully**

- **Command**: `python src/train.py --mode dev-only --data-dir data --test-end 10`
- **Result**: Training completed 3 epochs without errors
- **Checkpoints**: All 3 epoch checkpoints + best_model saved successfully
- **Logging**: Log file created with proper content

**Note**: Smoke test showed precision=0, recall=0, F1=0 because the tiny dataset has no gold links (all negative examples). This is expected behavior. Full training run will have positive examples and should show non-zero metrics.

---

## Usage Priority

1. Run local_setup.bat → dev dry-run (train.py --dev-only data/dev/2004-11-15_03)
2. Vast.ai: vast_setup.sh → full train (3-5hrs on RTX 4090)

Copy code from this MD to create files.

---

## Notes

- **Python Version**: Updated to support Python 3.13 (previously required Python 3.12 or lower)
- **PyTorch**: Using CPU version for compatibility
- **transformers**: Updated from 4.44.0 to 5.5.4
- **sentence-transformers**: Updated from 3.1.1 to 5.4.1
- **tokenizers**: Now compatible with Python 3.13 (previously failed due to PyO3 limitation)
- **Data loader**: Renamed from `data.py` to `data_loader.py` for clarity
- **Unit tests**: Created in `tests/test_data_loader.py` with 15 passing tests
- **Model tests**: Created in `tests/test_model.py` with 22 passing tests (1 skipped - CUDA not available)

---

## Quick Setup Commands (from llm_instructions.md)

```bash
cd ~/irc-disentanglement

python3 -m venv .venv-irc-bert
source .venv-irc-bert/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio
pip install transformers datasets sentence-transformers accelerate
```

**Note**: This is for Linux/remote setup. For Windows, use the setup.bat script above.

---

## Common Commands

### Activate Environment
```bash
# Windows
call .venv-irc-bert\Scripts\activate.bat

# Linux/Mac
source .venv-irc-bert/bin/activate
```

### Run Training
```bash
# Dev-only (quick test)
python src/train.py --dev-only data/dev/2004-11-15_03 --test-end 100

# Minimal test
python src/train.py --mode dev-only --data-dir data --test-end 10

# Full training
python src/train.py --mode train --data-dir data
```

### Check Logs
```bash
# View latest log
tail -f logs/train_*.log

# View data loader log
tail -f logs/data_loader_*.log
```

### Check GPU Status
```bash
nvidia-smi
```

---

## Troubleshooting

### Python 3.13 Compatibility Issues

If you encounter `IndentationError` on import:
- Ensure Python version is 3.13.11+ (not 3.13.8)
- Or use Python 3.12 for CUDA support

### CUDA Not Available

If `torch.cuda.is_available()` returns False:
- Use CPU version for development
- For GPU training, install Python 3.12 and recreate venv
- Use CUDA 12.1 compatible PyTorch

### Slow Training on CPU

Expected behavior:
- Dev-only (254 messages): ~5-10 minutes per epoch
- Full training (68,000 messages): Several hours per epoch
- Use Vast.ai RTX 4090 for faster training (~3-5 hours total)

### Test Range Not Working

If `--test-end` parameter is ignored:
- Use `--mode dev-only` instead of `--mode test`
- Check logs for actual test_start/test_end values
- Verify data_loader.py is limiting pairs correctly

---

## File Renaming Notes

- `src/data.py` has been renamed to `src/data_loader.py` for clarity
- Unit tests are now in `tests/test_data_loader.py`
- All imports updated to use `data_loader` module

---

## Environment Variables

No special environment variables required. All configuration is handled via command-line arguments.

---

## Project Structure Reference

```
irc_dis_pytorch/
├── data/                  # Ubuntu IRC dataset
├── src/                   # PyTorch code
│   ├── data_loader.py     # Data loading (renamed from data.py)
│   ├── model.py           # Model definition
│   └── train.py           # Training script
├── tests/                 # Unit tests
│   ├── test_data_loader.py
│   └── test_model.py
├── logs/                  # Log files
├── context/               # Documentation (this folder)
│   ├── CONTEXT.md         # Research context & architecture
│   ├── PROGRESS.md        # Progress & status
│   └── INSTRUCTIONS.md    # Setup & usage (this file)
└── .venv-irc-bert/        # Python virtual environment
```

---

## Next Steps

1. ✅ Set up environment using setup.bat (Windows) or setup.sh (Linux)
2. ✅ Run tests to verify installation
3. ✅ Run dev-only training to test pipeline (smoke test completed)
4. ✅ Check logs for progress tracking
5. 🔄 Execute full training run (3 epochs on full dataset)
6. 🔄 Perform inference on all 10 dev files
7. 🔄 Evaluate using graph-eval.py to compute link-level P/R/F1
8. 🔄 Compare results with DyNet baseline (62.2-62.6% F1)

---

*Last updated: 2026-04-16*
