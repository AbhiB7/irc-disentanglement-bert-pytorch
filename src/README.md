# IRC Disentanglement - PyTorch BERT Implementation

This is the new PyTorch BERT-based implementation of IRC conversation disentanglement.

## Project Structure
- src/ - New PyTorch code (data.py, model.py, etc.)
- data/ - IRC dataset (unchanged)
- plans/ - Planning documents
- rchive/jkummerfield-original/ - Original DyNet implementation
- 	ools/ - Evaluation scripts (kept for compatibility)

## Setup
1. Run setup.bat (Windows) or setup.sh (Linux) to install dependencies
2. Activate virtual environment: .venv-irc-bert\Scripts\activate
3. Run training: python train.py --help

## Verification
- Data loader should match original format from rchive/jkummerfield-original/src/disentangle.py
- Output format should be compatible with 	ools/evaluation/graph-eval.py
