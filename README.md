# CropXR Example: TCGA Gene Expression Autoencoder

This repository demonstrates machine learning analysis on cancer genomics data using PyTorch deep learning models. The project focuses on training autoencoders to learn meaningful representations from gene expression data derived from The Cancer Genome Atlas (TCGA).

## About TCGA

The Cancer Genome Atlas (TCGA) is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. TCGA data includes:

- **Gene expression profiles** from RNA sequencing
- **Clinical metadata** including patient demographics, treatment history, and outcomes
- **Multi-omics data** including copy number variations, methylation, and microRNA expression

This rich dataset enables researchers to discover molecular subtypes, identify biomarkers, and understand cancer biology at unprecedented scale.

TCGA is publically available. We use the processed output data of another repository: https://github.com/brmprnk/LB2292/tree/main/project2/data. For questions about the data, or to request a copy of the processed data, please contact t.verlaan@tudelft.nl ;
This script uses only the gene expression and clinical data. 
## What This Script Does

The main script (`src/main.py`) implements a deep learning pipeline that:
1. Loads the TCGA gene expression data
2. Trains an autoencoder model, using crossvalidation and early stopping
3. Generates UMAP visualizations of the learned representations


## Quick Start

### Local Development

```bash
# Install dependencies
pixi install

# Run with default parameters
pixi run python src/main.py

# Run with custom parameters
pixi run python src/main.py --epochs 500 --latent-dim 64 --batch-size 32
```

### Using Apptainer (Recommended for HPC)

Build the container:
```bash
apptainer build container.sif apptainer_v0.1.def
```

Run the analysis:
```bash
apptainer exec --writable-tmpfs --nv --pwd /opt/app --bind ./data/:data/ ./container.sif \
    pixi run python src/main.py [arguments]
```



## Configuration Options

All hyperparameters can be configured via command-line arguments. Run `pixi run python src/main.py --help` to see all options.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--latent-dim` | 128 | Size of the learned representation |
| `--hidden-dims` | 512 256 | Hidden layer sizes (encoder/decoder) |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 0.001 | Adam optimizer learning rate |
| `--n-splits` | 5 | Cross-validation folds |
| `--dropout` | 0.1 | Dropout rate for regularization |

### Example Commands

```bash
# Quick test run
python src/main.py --epochs 10 --n-splits 2

# Change model parameters
python src/main.py --latent-dim 64 --hidden-dims 256 128 --dropout 0.2

# Custom output directory
python src/main.py --output-dir results/experiment_1
```

## Output Files

The script generates:
- **`cv_loss.png`** - Cross-validation training curves
- **`umap_embeddings.png`** - 2D visualization of learned tumor representations

## Data Structure

Expected data files in `data/` directory:
- **`expression.pkl`** - Gene expression matrix (samples × genes)  
- **`clinical.csv`** - Clinical metadata with tumor tissue site annotations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn, pandas, matplotlib, seaborn
- UMAP-learn for dimensionality reduction
- Optional: CUDA-compatible GPU for acceleration