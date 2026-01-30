# MiniLLM: GPT from Scratch 

## Overview
**MiniLLM** is a minimal, educational implementation of a Generative Pre-trained Transformer (GPT) model, built entirely from scratch using PyTorch. 

The goal of this project is to demystify Large Language Models by implementing the core architecture described in the "Attention Is All You Need" paper, including:
- Multi-Head Self-Attention mechanisms
- Positional Embeddings
- Transformer Blocks (LayerNorm, FeedForward)
- Text generation loop

## Key Features
* **Tokenizer:** Character-level/Sub-word tokenization pipeline.
* **Transformer Architecture:** Pure implementation of encoder/decoder blocks without using `nn.Transformer`.
* **Training Loop:** Custom training loop with cross-entropy loss optimization.
* **Generation:** Autoregressive text generation capabilities.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/tejavardhan1/minillm.git](https://github.com/tejavardhan1/minillm.git)
   cd minillm
