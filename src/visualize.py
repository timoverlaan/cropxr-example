#!/usr/bin/env python3
"""
Data visualization and analysis script for CropXR project.
Provides utilities to explore and visualize the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


def plot_data_distribution(df, save_path="data_distribution.png"):
    """Plot distribution of features and target variable."""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Calculate number of subplots needed
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data distribution plot saved to {save_path}")


def plot_correlation_matrix(df, save_path="correlation_matrix.png"):
    """Plot correlation matrix of numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to {save_path}")


def plot_target_distribution(y, save_path="target_distribution.png"):
    """Plot distribution of target variable."""
    plt.figure(figsize=(8, 6))
    
    if len(np.unique(y)) <= 10:  # Categorical target
        counts = np.bincount(y)
        labels = [f'Class {i}' for i in range(len(counts))]
        plt.bar(labels, counts, alpha=0.7, edgecolor='black')
        plt.title('Target Variable Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
    else:  # Continuous target
        plt.hist(y, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Target Variable Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Target distribution plot saved to {save_path}")


def plot_pca_analysis(X, y, save_path="pca_analysis.png"):
    """Perform and visualize PCA analysis."""
    # Perform PCA
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Explained variance ratio
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title('PCA: Cumulative Explained Variance')
    ax1.grid(True)
    
    # Scree plot
    ax2.plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
             pca.explained_variance_ratio_[:20], 'ro-')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA: Scree Plot (First 20 Components)')
    ax2.grid(True)
    
    # 2D PCA scatter plot
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax3.set_title('PCA: First Two Principal Components')
    plt.colorbar(scatter, ax=ax3, label='Target')
    
    # Feature importance for first PC
    if X.shape[1] <= 20:  # Only show if manageable number of features
        feature_importance = np.abs(pca.components_[0])
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        ax4.bar(range(len(feature_importance)), feature_importance)
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Absolute Loading')
        ax4.set_title('Feature Importance in First Principal Component')
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Too many features\nto display loadings', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Feature Loadings (Too many to display)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA analysis plot saved to {save_path}")


def plot_tsne_visualization(X, y, save_path="tsne_visualization.png", 
                           perplexity=30, n_iter=1000):
    """Create t-SNE visualization of the data."""
    # Limit sample size for t-SNE if dataset is large
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
        print(f"Using random sample of 5000 points for t-SNE")
    else:
        X_sample = X
        y_sample = y
    
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Data')
    plt.colorbar(scatter, label='Target')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def visualize_images(image_dir, save_path="sample_images.png", n_samples=16):
    """Visualize a sample of images from the dataset."""
    image_path = Path(image_dir)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Sample random images
    n_samples = min(n_samples, len(image_files))
    sampled_files = np.random.choice(image_files, n_samples, replace=False)
    
    # Create subplot grid
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i, img_path in enumerate(sampled_files):
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(img_path.name, fontsize=8)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Hide empty subplots
    for j in range(len(sampled_files), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Sample Images from Dataset ({len(image_files)} total images)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample images visualization saved to {save_path}")


def generate_data_report(data_dir, output_dir="visualizations"):
    """Generate comprehensive data analysis report."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Generating data analysis report...")
    
    # Load tabular data if available
    csv_path = data_path / "data.csv"
    if csv_path.exists():
        print("Analyzing tabular data...")
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Save basic info
        with open(output_path / "data_summary.txt", "w") as f:
            f.write("Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes) + "\n\n")
            f.write("Basic Statistics:\n")
            f.write(str(df.describe()) + "\n")
        
        # Generate plots
        plot_data_distribution(df, output_path / "data_distribution.png")
        plot_correlation_matrix(df, output_path / "correlation_matrix.png")
        
        if 'target' in df.columns:
            plot_target_distribution(df['target'], output_path / "target_distribution.png")
    
    # Load processed data if available
    try:
        X = np.load(data_path / "X_train.npy")
        y = np.load(data_path / "y_train.npy")
        
        print("Analyzing processed data...")
        plot_pca_analysis(X, y, output_path / "pca_analysis.png")
        plot_tsne_visualization(X, y, output_path / "tsne_visualization.png")
        
    except FileNotFoundError:
        print("Processed data not found. Run preprocessing first for advanced analysis.")
    
    # Visualize images if available
    image_dir = data_path / "images"
    if image_dir.exists():
        print("Analyzing images...")
        visualize_images(image_dir, output_path / "sample_images.png")
    
    print(f"Data analysis report generated in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze CropXR data")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing data")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--report", action="store_true",
                       help="Generate full data analysis report")
    
    args = parser.parse_args()
    
    if args.report:
        generate_data_report(args.data_dir, args.output_dir)
    else:
        print("Use --report flag to generate comprehensive analysis")
        print("Example: python src/visualize.py --data-dir data --output-dir viz --report")


if __name__ == "__main__":
    main()