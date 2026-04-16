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
python src/train.py --epochs 10 --batch_size 16 --lr 2e-5
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

See LICENSE.md for details.
