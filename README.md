#  Transformers From Scratch: A Deep Dive into LLM Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains a modular, step-by-step implementation of the **Transformer** architecture as described in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). 

The goal of this project is to demystify the "black box" of Large Language Models (LLMs) by building every component—from the tokenizer to the attention heads—using only **PyTorch** and Python.

## Project Series Roadmap

The project is structured into logical phases, reflecting the flow of a modern NLP pipeline:

1.  **Phase 1: Tokenization & Embeddings**
    * Building a custom `SimpleTokenizer` with `<|unk|>` and `<|endoftext|>` support.
    * Turning integer IDs into dense, trainable vectors via `InputEmbeddings`.
2.  **Phase 2: Positional Encoding**
    * Injecting sequence order using Sinusoidal functions.
    * Implementing modern **Rotary Positional Embeddings (RoPE)** used in Llama-3.
3.  **Phase 3: The Attention Mechanism**
    * Scaled Dot-Product Attention math.
    * Multi-Head Attention (MHA) for parallel context processing.
4.  **Phase 4: Encoder & Decoder Architecture**
    * Layer Normalization and Residual Connections.
    * Position-wise Feed-Forward Networks (FFN).
5.  **Phase 5: Training & Inference**
    * Training the model on the *Alice in Wonderland* corpus.
    * Autoregressive text generation.

## Key Features

- **From-Scratch Implementation:** No `transformers` library imports. We use raw PyTorch for a deeper understanding. except the places where we are trying to copy the GPT architecture
- **Robust Tokenization:** Includes a custom regex-based tokenizer that handles out-of-vocabulary words.
- **Dimensionality Transparency:** Every tensor operation is commented with its shape (e.g., `[batch_size, seq_len, d_model]`) to help track the math.
- **Kaggle Optimized:** Designed to run efficiently in Kaggle or Colab environments.

##  Getting Started

### Prerequisites
```bash
pip install torch numpy
