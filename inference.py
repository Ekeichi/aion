"""
Inference and visualization script for JEPA model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from config import Config
from jepa_model import JEPAModel
from dataset import AthleteDataset


def load_model(model_path, config, device):
    """Load trained model"""
    model = JEPAModel(config).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")

    return model


def predict_future_embedding(model, x_t, a_t, device):
    """
    Predict future state embedding given context and action

    Args:
        model: Trained JEPA model
        x_t: Context window (window_length, num_features)
        a_t: Action (num_action_features,)

    Returns:
        s_y_pred: Predicted embedding (enc_y_dim,)
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension
        x_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0).to(device)
        a_t = torch.tensor(a_t, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        s_y_pred = model.inference(x_t, a_t)

        return s_y_pred.squeeze(0).cpu().numpy()


def compute_embedding_similarity(model, x_t, a_t, y_t, device):
    """
    Compute similarity between predicted and actual embeddings

    Returns:
        cosine_similarity: Cosine similarity between s_y_pred and s_y
        mse: MSE in embedding space
    """
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0).to(device)
        a_t = torch.tensor(a_t, dtype=torch.float32).unsqueeze(0).to(device)
        y_t = torch.tensor(y_t, dtype=torch.float32).unsqueeze(0).to(device)

        # Get embeddings
        s_y_pred, s_y, loss = model(x_t, a_t, y_t)

        # Compute metrics
        s_y_pred = s_y_pred.squeeze(0)
        s_y = s_y.squeeze(0)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            s_y_pred.unsqueeze(0),
            s_y.unsqueeze(0)
        ).item()

        # MSE
        mse = loss.item()

        return cos_sim, mse


def visualize_embeddings(model, test_loader, device, num_samples=100, save_path=None):
    """
    Visualize predicted vs actual embeddings using PCA

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        num_samples: Number of samples to visualize
    """
    from sklearn.decomposition import PCA

    model.eval()

    # Collect embeddings
    s_y_preds = []
    s_y_actuals = []

    with torch.no_grad():
        for i, (x_t, a_t, y_t, mask_x) in enumerate(test_loader):
            if i * test_loader.batch_size >= num_samples:
                break

            x_t = x_t.to(device)
            a_t = a_t.to(device)
            y_t = y_t.to(device)
            mask_x = mask_x.to(device)

            s_y_pred, s_y, _ = model(x_t, a_t, y_t, mask_x=mask_x)

            s_y_preds.append(s_y_pred.cpu().numpy())
            s_y_actuals.append(s_y.cpu().numpy())

    # Concatenate
    s_y_preds = np.concatenate(s_y_preds, axis=0)[:num_samples]
    s_y_actuals = np.concatenate(s_y_actuals, axis=0)[:num_samples]

    # PCA to 2D
    all_embeddings = np.concatenate([s_y_preds, s_y_actuals], axis=0)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    pred_2d = embeddings_2d[:num_samples]
    actual_2d = embeddings_2d[num_samples:]

    # Plot
    plt.figure(figsize=(10, 8))

    plt.scatter(pred_2d[:, 0], pred_2d[:, 1],
                alpha=0.6, s=50, c='blue', label='Predicted', marker='o')
    plt.scatter(actual_2d[:, 0], actual_2d[:, 1],
                alpha=0.6, s=50, c='red', label='Actual', marker='x')

    # Draw lines connecting predicted to actual
    for i in range(min(50, num_samples)):  # Only first 50 to avoid clutter
        plt.plot([pred_2d[i, 0], actual_2d[i, 0]],
                [pred_2d[i, 1], actual_2d[i, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title('Predicted vs Actual Embeddings (PCA)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_test_set(model, test_loader, device):
    """
    Analyze model performance on test set

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    losses = []
    cos_sims = []

    with torch.no_grad():
        for x_t, a_t, y_t, mask_x in test_loader:
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            y_t = y_t.to(device)
            mask_x = mask_x.to(device)

            s_y_pred, s_y, loss = model(x_t, a_t, y_t, mask_x=mask_x)

            # Batch metrics
            losses.append(loss.item())

            # Cosine similarity per sample
            for i in range(s_y_pred.size(0)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    s_y_pred[i:i+1], s_y[i:i+1]
                ).item()
                cos_sims.append(cos_sim)

    metrics = {
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'mean_cosine_similarity': np.mean(cos_sims),
        'std_cosine_similarity': np.std(cos_sims),
        'min_cosine_similarity': np.min(cos_sims),
        'max_cosine_similarity': np.max(cos_sims)
    }

    return metrics


def main():
    config = Config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model_path = Path(config.MODEL_DIR) / 'best_model.pt'
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    model = load_model(model_path, config, device)
    print()

    # Load test data
    csv_path = Path(config.DATA_DIR) / 'synthetic_athlete_data.csv'
    from dataset import create_dataloaders
    _, _, test_loader = create_dataloaders(csv_path, config, batch_size=32)

    # Analyze test set
    print("Analyzing test set performance...")
    metrics = analyze_test_set(model, test_loader, device)

    print("\n=== Test Set Metrics ===")
    print(f"Mean Loss (MSE):             {metrics['mean_loss']:.6f} ± {metrics['std_loss']:.6f}")
    print(f"Mean Cosine Similarity:      {metrics['mean_cosine_similarity']:.4f} ± {metrics['std_cosine_similarity']:.4f}")
    print(f"Cosine Similarity Range:     [{metrics['min_cosine_similarity']:.4f}, {metrics['max_cosine_similarity']:.4f}]")

    # Visualize embeddings
    print("\nGenerating embedding visualization...")
    viz_path = Path(config.LOGS_DIR) / 'embedding_visualization.png'
    visualize_embeddings(model, test_loader, device, num_samples=200, save_path=viz_path)

    print("\nInference complete!")


if __name__ == '__main__':
    main()
