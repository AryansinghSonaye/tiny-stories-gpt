# TinyStories GPT — Language Model from Scratch

A decoder-only Transformer language model implemented in PyTorch, based on the concepts and implementation approach presented in Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

The project trains a GPT-style language model from random initialization on a subset of the TinyStories dataset and provides an interactive script for text generation.

## Architecture

The model includes:

- Token embeddings and learned positional embeddings
- Causal multi-head self-attention
- Transformer blocks with residual connections
- Layer normalization
- GELU feed-forward networks
- A vocabulary projection head for next-token prediction

## Training

The current training configuration uses:

- **Dataset:** TinyStories, using the first 10,000 training examples
- **Tokenizer:** GPT-2 BPE tokenizer via `tiktoken`
- **Context length:** 256 tokens
- **Transformer layers:** 12
- **Attention heads:** 12
- **Embedding dimension:** 768
- **Optimizer:** AdamW
- **Learning rate:** 0.0004
- **Weight decay:** 0.1
- **Training duration:** 5 epochs
- **Train/validation split:** 90% / 10% of the concatenated text

These are the settings in the current implementation, not claims of optimal performance.

## Project Structure

```text
tiny-stories-gpt/
├── model.py       # Transformer architecture and weight-loading utilities
├── train.py       # Dataset preparation, training and loss evaluation
├── chat.py        # Autoregressive text generation
├── loss-plot.pdf  # Training and validation loss plot
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Use Python 3.10 or later and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your system and CUDA setup if GPU acceleration is desired.

## Train the Model

From the project directory, run:

```bash
python train.py
```

The script downloads the TinyStories dataset through Hugging Face Datasets, trains the model, evaluates training and validation loss, saves the model state, and generates a loss plot.

## Generate Text

After training has produced `TinyStoryPunyGPT.pth`, run:

```bash
python chat.py
```

Enter a prompt to generate text. Type `exit` to quit.

## Limitations

This is an educational, small-scale language-model training project. Training uses a limited subset of TinyStories and only five epochs. Generated text quality and generalization are therefore limited and should be evaluated empirically.

The model is trained from random initialization; it is not a pretrained GPT-2 checkpoint.

## Acknowledgements

This project follows the educational approach of Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

- Book and reference implementation: https://github.com/rasbt/LLMs-from-scratch
- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
