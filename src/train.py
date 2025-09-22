#!/usr/bin/env python3
"""
Main training script for CropXR project.
Trains a PyTorch model for crop classification/regression tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import os


class CropDataset(Dataset):
    """Custom dataset for CropXR data."""
    
    def __init__(self, X, y=None, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        
        if self.y is not None:
            return x, self.y[idx]
        return x


class CropMLP(nn.Module):
    """Multi-layer perceptron for crop data classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.2):
        super(CropMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CropCNN(nn.Module):
    """Convolutional Neural Network for image-based crop classification."""
    
    def __init__(self, num_classes=2, input_channels=3):
        super(CropCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Trainer:
    """Training class for crop models."""
    
    def __init__(self, model, device, log_dir="runs"):
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(targets, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(targets, predictions)
        
        return avg_loss, accuracy, predictions, targets
    
    def train(self, train_loader, val_loader, optimizer, criterion, 
              num_epochs=100, patience=10, save_path="best_model.pth"):
        """Complete training loop with early stopping."""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_metrics(self, save_path="training_metrics.png"):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Training metrics plot saved to {save_path}")


def load_data(data_dir):
    """Load preprocessed data."""
    data_path = Path(data_dir)
    
    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="Train CropXR model")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                       help="Directory containing processed data")
    parser.add_argument("--model-type", type=str, choices=["mlp", "cnn"], default="mlp",
                       help="Type of model to train")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        print("Loading data...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_dir)
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Create datasets and dataloaders
        train_dataset = CropDataset(X_train, y_train)
        val_dataset = CropDataset(X_val, y_val)
        test_dataset = CropDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        if args.model_type == "mlp":
            model = CropMLP(input_dim=input_dim, num_classes=num_classes)
        else:
            # For CNN, assume image data is reshaped appropriately
            model = CropCNN(num_classes=num_classes)
        
        print(f"Model: {model}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Create trainer
        trainer = Trainer(model, device, log_dir=output_dir / "tensorboard")
        
        # Train model
        print("Starting training...")
        train_losses, val_losses, train_accs, val_accs = trainer.train(
            train_loader, val_loader, optimizer, criterion,
            num_epochs=args.epochs, patience=args.patience,
            save_path=output_dir / "best_model.pth"
        )
        
        # Plot metrics
        trainer.plot_metrics(output_dir / "training_metrics.png")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_preds, test_targets = trainer.validate(test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(test_targets, test_preds))
        
        # Save results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accs,
            "val_accuracies": val_accs,
            "model_type": args.model_type,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device)
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining complete! Results saved to {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please run preprocessing first.")
        print(f"Expected files in {args.data_dir}:")
        print("- X_train.npy, X_val.npy, X_test.npy")
        print("- y_train.npy, y_val.npy, y_test.npy")
        print(f"\nRun: python src/preprocess.py --data-dir your_data_directory")


if __name__ == "__main__":
    main()