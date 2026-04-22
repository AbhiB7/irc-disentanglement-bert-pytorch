# IRC Conversation Disentanglement (PyTorch)

A BERT-based CrossEncoder with handcrafted features for IRC message linking and conversation disentanglement.

## Project Structure

```
.
├── src/
│   ├── data_loader.py    # Dataset loading and preprocessing
│   ├── model.py          # CrossEncoder model definition
│   └── train.py          # Training script
├── tests/
│   ├── test_data_loader.py
│   └── test_model.py
├── data/                 # Training data (IRC conversations)
├── tools/                # Evaluation and preprocessing tools
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── .gitignore
├── setup.bat
├── create_tiny.py        # Create tiny dataset for testing
└── check_dependencies.py
```

## AI Context & Documentation

This project uses a strict three-file system in the `context/` directory to manage AI context and prevent documentation rot. **Do not mix their responsibilities.**

- **[`instructions.md`](context/INSTRUCTIONS.md)**: Stable behavioral rules for AI agents.
- **[`context.md`](context/CONTEXT.md)**: Stable project knowledge, research background, and technical invariants.
- **[`progress.md`](context/PROGRESS.md)**: Dynamic working state, recent completions, and next steps.

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers tqdm numpy
   ```

3. Download GloVe embeddings (if needed) and ensure data files are in `data/`

## Usage

### Training

```bash
python src/train.py --epochs 3 --batch_size 64 --learning-rate 5e-5
```

### Testing

```bash
python -m pytest tests/
```

## Features

- BERT-based CrossEncoder for message pair classification
- Handcrafted features (temporal, lexical, network)
- Support for Ubuntu and Channel-two IRC datasets

## License

See LICENSE.md for details.1

## Acknowledgements

https://github.com/jkkummerfeld/irc-disentanglement