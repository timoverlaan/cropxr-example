import pickle as pkl
import anndata as ad
import pandas as pd
import numpy as np
import umap
import argparse

from tqdm import tqdm, trange
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

print("CUDA available:", torch.cuda.is_available())


def parse_args():
    """Parse command line arguments for hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Train autoencoder on gene expression data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Minimum improvement for early stopping')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--inference-batch-size', type=int, default=128, help='Batch size for inference (UMAP generation)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer')
    
    # Model hyperparameters
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension size')
    parser.add_argument('--hidden-dims', type=int, nargs=2, default=[512, 256], 
                        help='Hidden layer dimensions (two values)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Cross-validation
    parser.add_argument('--n-splits', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    
    # UMAP parameters
    parser.add_argument('--umap-min-dist', type=float, default=0.5, help='UMAP min_dist parameter')
    parser.add_argument('--umap-n-neighbors', type=int, default=50, help='UMAP n_neighbors parameter')
    
    # Data paths
    parser.add_argument('--expression-data', type=str, default='data/expression.pkl', 
                        help='Path to expression data pickle file')
    parser.add_argument('--clinical-data', type=str, default='data/clinical.csv', 
                        help='Path to clinical data CSV file')
    
    # Output paths
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots')
    
    return parser.parse_args()


class MLPBlock(nn.Module):
    """Linear -> BatchNorm -> GELU -> Dropout -> Linear (residual)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        # BatchNorm expects [B, C]; handle 2D inputs
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h  # residual


class Autoencoder(nn.Module):
    """
    Slightly fancier AE:
      - Encoder/decoder with GELU, BatchNorm, Dropout
      - Residual block per hidden layer
      - Learned skip connection input->output (helps reconstruction)
      - Helper to compute MSE reconstruction loss
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=(512, 128), dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims

        # ----- Encoder -----
        self.enc_fc1 = nn.Linear(input_dim, h1)
        self.enc_bn1 = nn.BatchNorm1d(h1)
        self.enc_block1 = MLPBlock(h1, dropout)

        self.enc_fc2 = nn.Linear(h1, h2)
        self.enc_bn2 = nn.BatchNorm1d(h2)
        self.enc_block2 = MLPBlock(h2, dropout)

        self.enc_latent = nn.Linear(h2, latent_dim)

        # ----- Decoder -----
        self.dec_fc1 = nn.Linear(latent_dim, h2)
        self.dec_bn1 = nn.BatchNorm1d(h2)
        self.dec_block1 = MLPBlock(h2, dropout)

        self.dec_fc2 = nn.Linear(h2, h1)
        self.dec_bn2 = nn.BatchNorm1d(h1)
        self.dec_block2 = MLPBlock(h1, dropout)

        self.dec_out = nn.Linear(h1, input_dim)

        # Learned skip projection (input → output space)
        # self.skip = nn.Linear(input_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0)  # good for GELU
                nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.enc_fc1(x)
        x = self.enc_bn1(x)
        x = F.gelu(x)
        x = self.enc_block1(x)

        x = self.enc_fc2(x)
        x = self.enc_bn2(x)
        x = F.gelu(x)
        x = self.enc_block2(x)

        z = self.enc_latent(x)
        return z

    def decode(self, z):
        x = self.dec_fc1(z)
        x = self.dec_bn1(x)
        x = F.gelu(x)
        x = self.dec_block1(x)

        x = self.dec_fc2(x)
        x = self.dec_bn2(x)
        x = F.gelu(x)
        x = self.dec_block2(x)

        x = self.dec_out(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        # Add learned skip (acts like an identity shortcut if useful)
        out = recon   # + self.skip(x)
        return out, z

    @staticmethod
    def compute_loss(x, out):
        """
        Mean squared reconstruction error (returns scalar).
        Usage:
            out, _ = model(x)
            loss = model.compute_loss(x, out)
        """
        return F.mse_loss(out, x)


def load_and_prepare_data(args):
    """Load and prepare TCGA data for training."""
    print("Loading data...")
    
    # Load expression data
    data = pkl.load(open(args.expression_data, "rb"))
    data_pt = data[data["sample_type"] == "Primary Tumor"]
    data_pt = data_pt.drop_duplicates(subset=["patient_id"])

    # Load clinical metadata
    metadata = pd.read_csv(args.clinical_data, index_col=0)

    # Create AnnData object
    adata = ad.AnnData(data_pt.set_index("patient_id").drop(columns=["sample_type"]))
    adata.obs = metadata.loc[adata.obs_names].copy()
    
    print(f"Data loaded: {adata.n_obs} samples, {adata.n_vars} genes")
    
    return adata


def train_fold(model, train_loader, val_loader, optimizer, criterion, args, device):
    """Train a single fold and return the best validation loss and loss history."""
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    fold_loss_log = []

    for epoch in trange(args.epochs, desc="Training"):
        # Training phase
        model.train()
        for batch in train_loader:
            x_batch = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_recon, _ = model(x_batch)
            loss = criterion(x_recon, x_batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device, non_blocking=True)
                x_recon, _ = model(x_batch)
                loss = criterion(x_recon, x_batch)
                bs = x_batch.size(0)
                val_loss_sum += loss.item() * bs
                n += bs
        avg_val_loss = val_loss_sum / max(1, n)
        fold_loss_log.append(avg_val_loss)

        # Early stopping
        if (best_val - avg_val_loss) > args.min_delta:
            best_val = avg_val_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    return best_val, best_state, fold_loss_log


def generate_embeddings(model, X_np, args, device):
    """Generate embeddings in batches to avoid memory issues."""
    model.eval()
    batch_size = min(args.inference_batch_size, len(X_np))
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            end_idx = min(i + batch_size, len(X_np))
            X_batch = torch.tensor(X_np[i:end_idx], dtype=torch.float32, device=device)
            batch_embeddings = model.encode(X_batch).cpu().numpy()
            embeddings.append(batch_embeddings)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
    
    return np.vstack(embeddings)


def create_plots(cv_loss_log, embeddings, adata, args):
    """Create and save visualization plots."""
    # Cross-validation loss plot
    plt.figure(figsize=(10, 6))
    for fold, val_loss in enumerate(cv_loss_log):
        plt.plot(val_loss, label=f"Fold {fold+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Cross-Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.output_dir}/cv_loss.png", bbox_inches='tight', dpi=300)
    plt.close()

    # UMAP plot
    print("Computing UMAP...")
    umap_embed = umap.UMAP(
        n_components=2, 
        min_dist=args.umap_min_dist, 
        n_neighbors=args.umap_n_neighbors,
        random_state=args.random_seed
    )
    X_umap_embed = umap_embed.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_umap_embed[:, 0], 
        y=X_umap_embed[:, 1], 
        hue=adata.obs['tumor_tissue_site'], 
        s=15, 
        alpha=0.7
    )
    plt.title("UMAP of Learned Embeddings", fontsize=16)
    plt.xlabel("UMAP 1", fontsize=14)
    plt.ylabel("UMAP 2", fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Tumor Tissue Site')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/umap_embeddings.png", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("AUTOENCODER TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Training Parameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Early stopping patience: {args.patience}")
    print(f"  - Min delta: {args.min_delta}")
    print()
    print(f"Model Architecture:")
    print(f"  - Latent dimension: {args.latent_dim}")
    print(f"  - Hidden dimensions: {args.hidden_dims}")
    print(f"  - Dropout rate: {args.dropout}")
    print()
    print(f"Cross-validation:")
    print(f"  - Number of folds: {args.n_splits}")
    print(f"  - Random seed: {args.random_seed}")
    print()
    print(f"UMAP Parameters:")
    print(f"  - Min distance: {args.umap_min_dist}")
    print(f"  - Number of neighbors: {args.umap_n_neighbors}")
    print()
    print(f"Data and Output:")
    print(f"  - Expression data: {args.expression_data}")
    print(f"  - Clinical data: {args.clinical_data}")
    print(f"  - Output directory: {args.output_dir}")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Load and prepare data
    adata = load_and_prepare_data(args)
    
    # Show model summary
    model = Autoencoder(
        input_dim=adata.n_vars, 
        latent_dim=args.latent_dim, 
        hidden_dims=tuple(args.hidden_dims), 
        dropout=args.dropout
    )
    print("\nModel Architecture:")
    summary(model, input_size=(args.batch_size, adata.n_vars), device=torch.device("cpu"))

    # Setup device and cross-validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    X_np = adata.X.astype('float32')
    cv_loss_log = []

    # Cross-validation training
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        print(f"\nFold {fold+1}/{args.n_splits}")
        
        # Prepare data loaders
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=args.batch_size, shuffle=False)

        # Initialize model for this fold
        model = Autoencoder(
            input_dim=adata.n_vars, 
            latent_dim=args.latent_dim, 
            hidden_dims=tuple(args.hidden_dims), 
            dropout=args.dropout
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()

        # Train this fold
        best_val, best_state, fold_loss_log = train_fold(
            model, train_loader, val_loader, optimizer, criterion, args, device
        )

        # Restore best weights and record results
        if best_state is not None:
            model.load_state_dict(best_state)

        cv_loss_log.append(fold_loss_log)
        print(f"  Best validation loss: {best_val:.4f}")
        
        # Clear GPU memory after each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate final embeddings and visualizations
    print("\nGenerating final visualizations...")
    embeddings = generate_embeddings(model, X_np, args, device)
    create_plots(cv_loss_log, embeddings, adata, args)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Results saved to {args.output_dir}/:")
    print(f"   - cv_loss.png (cross-validation curves)")
    print(f"   - umap_embeddings.png (tumor visualization)")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()