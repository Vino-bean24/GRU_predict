# GRU_practice: Protein Sequence Modeling with PyTorch

## Overview

GRU_practice is a deep learning project for protein sequence modeling and generation, built with PyTorch. The project demonstrates how to use a Gated Recurrent Unit (GRU) based Recurrent Neural Network (RNN) to learn the statistical patterns of protein sequences and generate new sequences. This project is designed as a practical demo for beginners in deep learning and bioinformatics, and can serve as a portfolio piece for graduate school applications or job interviews.

## Features
- **Protein sequence tokenization and vocabulary building**
- **Custom PyTorch Dataset and DataLoader logic**
- **GRU-based RNN model for sequence prediction**
- **Training and validation loops with loss and perplexity tracking**
- **Sequence generation (auto-regressive prediction)**
- **Visualization of training/validation loss and perplexity**
- **Model saving and loading for inference**

## Project Structure
```
project/
  GRU_practice/
    model.py            # Main script: data processing, model, training, prediction
    protein_model.path  # Saved model parameters
    Figure_1.png        # Example figure (e.g., loss/perplexity curves)
    data/
      1000.fasta        # Training protein sequences (FASTA format)
      1000_1250.fasta   # Validation protein sequences (FASTA format)
```

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- numpy
- matplotlib

Install dependencies:
```bash
pip install torch numpy matplotlib
```

### Data Preparation
- Place your protein sequence files in FASTA format under `project/GRU_practice/data/`.
- Example files: `1000.fasta` (train), `1000_1250.fasta` (validation).

### Training
Run the main script to train the model:
```bash
python model.py --epochs 200 --train_batch_size 6400 --eval_batch_size 3200 --max_sql 35 --cuda
```
- Use `--cuda` to enable GPU training if available.
- All arguments are optional and have reasonable defaults.

### Inference / Sequence Generation
After training, you can generate new protein sequences:
```python
from model import RNN, Corpus, predict
import torch

# Load data and model
corpus = Corpus('data', {'train': 6400, 'valid': 3200}, 35)
nvoc, ninput, nhid, nlayers = len(corpus.vocabulary), 128, 256, 2
net = RNN(nvoc, ninput, nhid, nlayers)
net.load_state_dict(torch.load('protein_model.path', map_location='cpu'))
net.eval()

# Predict
initial_sequence = ['M', 'E', 'N', 'S', 'D']
predicted = predict(initial_sequence, 20, net, corpus, torch.device('cpu'), 35)
print(''.join(predicted))
```

## Example Results
- Training and validation loss/perplexity curves are saved as `Figure_1.png`.
- The model can generate plausible protein-like sequences given a short initial fragment.

## Applications
- Demonstrates sequence modeling for bioinformatics
- Can be extended for protein design, mutation effect prediction, or other sequence tasks
- Serves as a template for RNN/GRU-based modeling in other domains

## Why This Project?
This project was developed as a hands-on demo for learning deep learning and sequence modeling, and as a showcase for graduate school or job applications. It highlights:
- End-to-end workflow: data processing, model building, training, evaluation, and inference
- Clean, well-commented code suitable for beginners
- Practical application in computational biology

## License
This project is open source and available under the MIT License.

## Contact
For questions or collaboration, please contact: [Your Name] ([your.email@example.com])
