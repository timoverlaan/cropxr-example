#!/usr/bin/env python3
"""
Data preprocessing script for CropXR project.
Handles data loading, cleaning, and preparation for ML training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


def load_image_data(image_dir: Path, target_size: tuple = (224, 224)):
    """
    Load and preprocess images from a directory.
    
    Args:
        image_dir: Path to directory containing images
        target_size: Target size for resizing images (width, height)
    
    Returns:
        List of preprocessed image arrays
    """
    images = []
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            # Load image using PIL
            img = Image.open(img_path).convert('RGB')
            # Resize image
            img = img.resize(target_size)
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return np.array(images)


def load_tabular_data(csv_path: Path):
    """
    Load and preprocess tabular data.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Preprocessed DataFrame
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating sample data.")
        # Create sample data for demonstration
        data = {
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.randint(0, 2, 1000)
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Sample data saved to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
    
    return df


def preprocess_tabular_data(df: pd.DataFrame):
    """
    Preprocess tabular data with scaling and encoding.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Preprocessed features and targets
    """
    # Separate features and target
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        X = df
        y = None
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Scale numerical features
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion for test set
        val_size: Proportion for validation set from remaining data
        random_state: Random seed
    
    Returns:
        Split datasets
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="Preprocess CropXR data")
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Directory containing raw data")
    parser.add_argument("--output-dir", type=str, default="processed_data",
                       help="Directory to save processed data")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                       help="Target image size (width height)")
    
    args = parser.parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Starting data preprocessing...")
    
    # Process tabular data
    csv_path = data_dir / "data.csv"
    print(f"Loading tabular data from {csv_path}")
    df = load_tabular_data(csv_path)
    
    print("Preprocessing tabular data...")
    X, y, scaler = preprocess_tabular_data(df)
    
    if y is not None:
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Save processed data
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_val.npy", y_val)
        np.save(output_dir / "y_test.npy", y_test)
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Process images if directory exists
    image_dir = data_dir / "images"
    if image_dir.exists():
        print(f"Loading images from {image_dir}")
        images = load_image_data(image_dir, tuple(args.image_size))
        np.save(output_dir / "images.npy", images)
        print(f"Processed {len(images)} images")
    
    print(f"Preprocessing complete! Data saved to {output_dir}")


if __name__ == "__main__":
    main()