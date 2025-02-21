# Next-Word-Prediction-for-University-FAQs-Section


This repository contains a deep learning model implemented in PyTorch for next-word prediction using an LSTM-based neural network. The model is trained on a tokenized dataset, where input sequences predict the next word in a sentence.

## Features
- Uses an LSTM-based neural network for sequence prediction.
- Implements text tokenization and numerical encoding.
- Supports batch training with padded sequences.
- Uses PyTorch for deep learning and model training.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install torch numpy
```

## Usage

1. Prepare your dataset and ensure it follows the tokenized format.
2. Train the model using the provided notebook.
3. Evaluate the model's performance using the given metrics.

## Model Architecture
- **Embedding Layer**: Converts tokens into dense vectors.
- **LSTM Layer**: Captures temporal dependencies in text.
- **Fully Connected Output Layer**: Maps LSTM outputs to vocabulary size.

## Metrics

| Metric | Value / Shape | Description |
|--------|--------------|-------------|
| Vocabulary Size (len(vocab)) | 356 | Unique tokens in the dataset. |
| Number of Sentences (len(input_numerical_sentences)) | 59 | Total input sentences processed. |
| Total Training Sequences (len(training_sequence)) | 856 | Number of training examples created. |
| Longest Sequence for Padding | 112 | Maximum sequence length for padding. |
| Training Data Shape (training_sequence) | (856, 112) | Total sequences padded to 112 tokens. |
| Input Tensor Shape (X.shape) | (856, 111) | X contains all input sequences (excluding the last token). |
| Target Tensor Shape (y.shape) | (856, ) | y contains the labels (next-word predictions). |
| Batch Size | 32 | Number of sequences per batch during training. |
| Embedding Layer Output | (32, 111, 100) | Batch size 32, sequence length 111, embedding dim 100. |
| LSTM Input Shape | (32, 111, 100) | Embedding vectors passed to LSTM. |
| LSTM Hidden State Shape | (1, 32, 150) | LSTM's final hidden state (1 layer, 32 batches, 150 units). |
| LSTM Cell State Shape | (1, 32, 150) | LSTM's final cell state for learning long-term dependencies. |
| Output Layer Shape | (32, 356) | Final prediction layer (batch size × vocab size). |


![Image](https://github.com/user-attachments/assets/1fdeaa7f-4b67-4fda-9b5b-82539ef088db)


