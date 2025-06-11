# SAE Memorization Detection Pipeline

This repository contains the core pipeline for detecting memorization in language models using Sparse Autoencoders (SAEs). The pipeline processes data from the Carlini memorization dataset, extracts SAE activations, and trains probes to detect memorized content.

## Overview

The pipeline consists of four main steps:

1. **Memorization Detection**: Identify which sequences are memorized by the model
2. **Dataset Balancing**: Create a balanced dataset controlling for confounding factors
3. **Activation Extraction**: Extract SAE activations for the balanced dataset
4. **Probe Training**: Train classifiers to detect memorization from SAE features

## Prerequisites

- **Carlini Dataset**: You'll need sequences from the Carlini memorization dataset with duplication counts, prefix NLL, and rare token rate pre-computed
  - A small sample dataset (`sample_dataset_100.jsonl`) with 100 examples is included for testing
- **SAE Checkpoints**: Pre-trained SAEs compatible with your model (e.g., from HuggingFace)
- **GPU**: Recommended for steps 1, 3, and 4

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Main requirements:
- Python 3.8+
- PyTorch 2.0+
- Transformers
- scikit-learn
- pyarrow
- pandas
- numpy

## Quick Start with Sample Dataset

To quickly test the pipeline with the included sample dataset (100 examples):

```bash
# Skip steps 1-2 since sample_dataset_100.jsonl already has labels
# Step 3: Extract activations
python extract_sae_activations.py \
    --data sample_dataset_100.jsonl \
    --model EleutherAI/pythia-2.8b \
    --layers 15 \
    --sae_repo "your-sae-repo" \
    --out_dir sample_activations \
    --batch 8

# Step 4: Train probe
python train_memorization_probe.py \
    --parquet_dir sample_activations \
    --layer 15 \
    --n_each 50 \
    --hidden 512 \
    --epochs 5
```

## Full Pipeline Steps

### Step 1: Generate Memorization Labels

Detect which sequences from the Carlini dataset are memorized by generating continuations and checking for exact matches.

```bash
python detect_memorization.py input_sequences.jsonl \
    --out memorized_labeled.jsonl \
    --prefix 96 \
    --gen 32 \
    --pos_thr 0.99 \
    --neg_thr 0.1 \
    --batch 16
```

**Key parameters:**
- `--prefix`: Number of tokens to use as prompt (default: 32)
- `--gen`: Number of tokens to generate (default: 64)
- `--pos_thr`: Threshold for memorization detection (default: 0.8)
- `--neg_thr`: Threshold for non-memorization (default: 0.1)

### Step 2: Create Balanced Dataset

Balance memorized and non-memorized examples while controlling for confounding factors (duplication count, prefix NLL, rare token rate).

```bash
python create_balanced_dataset.py memorized_labeled.jsonl \
    --out_jsonl balanced_dataset.jsonl \
    --stats_csv matching_stats.csv \
    --target_each 6900 \
    --min_total 12000
```

**Output:**
- `balanced_dataset.jsonl`: Balanced dataset with equal positive/negative samples
- `matching_stats.csv`: Statistics on matching quality

### Step 3: Extract SAE Activations

Extract SAE activations for all sequences in the balanced dataset.

```bash
python extract_sae_activations.py \
    --data balanced_dataset.jsonl \
    --model EleutherAI/pythia-2.8b \
    --layers 0 1 2 15 \
    --sae_repo "your-sae-repo" \
    --out_dir sae_activations \
    --batch 32 \
    --pool mean
```

**Key parameters:**
- `--layers`: Which transformer layers to extract (e.g., 0 1 2 15)
- `--sae_repo`: HuggingFace repo containing trained SAEs
- `--pool`: Pooling method for sequence features (mean/max/last/both)
- `--out_dir`: Directory to save Parquet files

**Output:** Parquet files with structure `L{layer}_{shard}.parquet`

### Step 4: Train Memorization Probe

Train an MLP probe to detect memorization from SAE activations.

```bash
python train_memorization_probe.py \
    --parquet_dir sae_activations \
    --layer 15 \
    --n_each 5000 \
    --hidden 1024 \
    --epochs 8 \
    --batch 512
```

**Key parameters:**
- `--layer`: Which layer's activations to use
- `--n_each`: Number of examples per class
- `--hidden`: Hidden layer size for MLP
- `--scale`: Feature scaling method (standard/maxabs/none)

## Alternative Analysis

For baseline comparison without SAE features:

```bash
python baseline_logistic_regression.py balanced_dataset.jsonl \
    --features dup_count prefix_nll rare_rate
```

## Data Format

### Input Format (Carlini dataset)
```json
{
    "text": "sequence text",
    "dup_count": 1234,
    "prefix_nll": 12.5,
    "rare_rate": 0.15
}
```

### Matched Dataset Format
```json
{
    "label": "mem",
    "text": "sequence text",
    "dup_count": 1234,
    "prefix_nll": 12.5,
    "rare_rate": 0.15,
    "domain": "code",
    "match_ratio": 0.95,
    "match_consec": 45
}
```

## Notes

- The pipeline assumes Pythia-2.8B model by default
- SAE checkpoints should be compatible with the model architecture
- GPU recommended for Steps 1, 3, and 4
- Adjust batch sizes based on available GPU memory