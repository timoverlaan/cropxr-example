# CropXR Example Project

Data and Deep Learning model example for CropXR project using **pixi** for dependency management.

## Features

- **PyTorch-based ML pipeline** for crop classification/regression
- **Pixi package management** with conda environments
- **Modular data processing scripts** for preprocessing, augmentation, and visualization
- **Comprehensive training script** with early stopping, tensorboard logging, and metrics visualization
- **Sample data generation** for quick testing and development

## Setup

### Prerequisites

- [Pixi](https://pixi.sh/) package manager installed on your system

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cropxr-example
```

2. Install dependencies using pixi:
```bash
pixi install
```

3. Activate the environment:
```bash
pixi shell
```

## Project Structure

```
├── src/                     # Source code directory
│   ├── train.py            # Main training script with PyTorch models
│   ├── preprocess.py       # Data preprocessing and splitting
│   ├── augment.py          # Image augmentation utilities
│   └── visualize.py        # Data visualization and analysis
├── data/                   # Raw data directory (created automatically)
├── processed_data/         # Processed data directory (created by preprocessing)
├── models/                 # Trained models and results
├── pixi.toml              # Pixi configuration and dependencies
└── README.md              # This file
```

## Usage

### Quick Start

1. **Generate sample data and run preprocessing:**
```bash
pixi run preprocess --data-dir data --output-dir processed_data
```

2. **Train a model:**
```bash
pixi run train --data-dir processed_data --epochs 10 --batch-size 32 --output-dir models
```

3. **Visualize data (optional):**
```bash
pixi run visualize --data-dir data --output-dir visualizations --report
```

### Available Pixi Tasks

- `pixi run train` - Train a PyTorch model
- `pixi run preprocess` - Preprocess and split data
- `pixi run augment` - Apply data augmentation
- `pixi run visualize` - Generate data visualizations

### Detailed Usage

#### Data Preprocessing

The preprocessing script handles both tabular and image data:

```bash
pixi run preprocess --data-dir data --output-dir processed_data --image-size 224 224
```

Features:
- Automatic sample data generation if no data is found
- Support for CSV files and image directories
- Feature scaling and encoding
- Train/validation/test splitting
- Image resizing and normalization

#### Model Training

Train various model architectures:

```bash
# Train MLP for tabular data
pixi run train --model-type mlp --epochs 100 --lr 0.001 --batch-size 32

# Train CNN for image data (when available)
pixi run train --model-type cnn --epochs 50 --lr 0.0001 --batch-size 16
```

Features:
- Multi-layer perceptron (MLP) and Convolutional Neural Network (CNN) models
- Early stopping with patience
- TensorBoard logging
- Automatic model checkpointing
- Comprehensive evaluation metrics

#### Data Augmentation

Apply various augmentation techniques to images:

```bash
pixi run augment --input-dir data/images --output-dir data/augmented --augmentations 3
```

Features:
- Rotation, brightness, contrast adjustments
- Horizontal flipping
- Noise addition and blurring
- Configurable augmentation parameters

#### Data Visualization

Generate comprehensive data analysis reports:

```bash
pixi run visualize --data-dir data --output-dir visualizations --report
```

Features:
- Distribution plots for all features
- Correlation matrices
- PCA and t-SNE visualizations
- Sample image displays
- Comprehensive data summary reports

## Dependencies

The project uses the following main dependencies (managed by pixi):

- **PyTorch** - Deep learning framework
- **torchvision** & **torchaudio** - PyTorch companion libraries
- **NumPy** & **Pandas** - Data manipulation
- **scikit-learn** - Traditional ML algorithms and metrics
- **Matplotlib** & **Seaborn** - Visualization
- **OpenCV** & **Pillow** - Image processing
- **TensorBoard** - Training visualization
- **Jupyter** - Interactive development
- **tqdm** - Progress bars

## CUDA Support

The project supports CUDA acceleration when available. To use GPU training:

1. Ensure you have CUDA-compatible hardware and drivers
2. The training script automatically detects CUDA availability
3. Use `--device cuda` to force GPU usage or `--device cpu` for CPU-only training

## Example Workflow

1. **Prepare your data:** Place CSV files and images in the `data/` directory
2. **Preprocess:** Run `pixi run preprocess` to clean and split your data
3. **Visualize:** Run `pixi run visualize --report` to understand your data
4. **Augment (optional):** Run `pixi run augment` to increase dataset size
5. **Train:** Run `pixi run train` with desired parameters
6. **Evaluate:** Check results in the `models/` directory

## Development

To add new dependencies:

```bash
pixi add <package-name>
```

To update dependencies:

```bash
pixi update
```

## License

This project is part of the CropXR research initiative.

## Contributing

Please follow the existing code structure and ensure all scripts work with the pixi environment before submitting changes.