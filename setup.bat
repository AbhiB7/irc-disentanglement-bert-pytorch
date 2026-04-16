@echo off
REM Setup script for IRC Disentanglement PyTorch BERT project on Windows
REM Assumes Python 3.12+ installed, NVIDIA GPU with CUDA 12.1 compatible driver

if not exist ".venv-irc-bert" (
    python -m venv .venv-irc-bert
)
call .venv-irc-bert\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch (CPU version for compatibility)
pip install torch torchvision torchaudio

REM Core ML dependencies (updated for Python 3.13 compatibility)
pip install transformers==5.5.4 datasets sentence-transformers==5.4.1 accelerate

REM Additional utils
pip install scikit-learn numpy pandas tqdm

echo Setup complete. Run 'call .venv-irc-bert\Scripts\activate.bat' to activate.
