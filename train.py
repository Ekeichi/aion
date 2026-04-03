"""
Training script for JEPA Digital Twin model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from jepa_model import JEPAModel, count_parameters
from dataset import create_dataloaders
from data_generator import DataGenerator


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for x_t, a_t, y_t, mask_x in pbar:
        # Move to device
        x_t = x_t.to(device)
        a_t = a_t.to(device)
        y_t = y_t.to(device)
        mask_x = mask_x.to(device)

        # Forward pass
        optimizer.zero_grad()
        s_y_pred, s_y, loss = model(x_t, a_t, y_t, mask_x=mask_x)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_t, a_t, y_t, mask_x in tqdm(loader, desc='Validation'):
            # Move to device
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            y_t = y_t.to(device)
            mask_x = mask_x.to(device)

            # Forward pass
            s_y_pred, s_y, loss = model(x_t, a_t, y_t, mask_x=mask_x)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE in Latent Space)', fontsize=12)
    plt.title('JEPA Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    # Configuration
    config = Config()

    # Setup paths
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    Path(config.LOGS_DIR).mkdir(exist_ok=True)

    # Check for data, generate if needed
    csv_path = Path(config.DATA_DIR) / 'synthetic_athlete_data.csv'
    if not csv_path.exists():
        print("No data found. Generating synthetic dataset...")
        generator = DataGenerator(config)
        generator.generate_dataset()
        print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(csv_path, config)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create model
    print("Creating JEPA model...")
    model = JEPAModel(config).to(device)
    print(f"Total parameters: {count_parameters(model):,}")
    print()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Training loop
    print("Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Log
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = Path(config.MODEL_DIR) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, best_model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

        print()

        # Plot curves every 10 epochs
        if epoch % 10 == 0:
            plot_path = Path(config.LOGS_DIR) / f'training_curves_epoch{epoch}.png'
            plot_training_curves(train_losses, val_losses, plot_path)

    # Training complete
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Final plots
    plot_path = Path(config.LOGS_DIR) / 'training_curves_final.png'
    plot_training_curves(train_losses, val_losses, plot_path)

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'config': config.__dict__
    }
    history_path = Path(config.LOGS_DIR) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Test on test set
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")

    # Save final model
    final_model_path = Path(config.MODEL_DIR) / 'final_model.pt'
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'config': config.__dict__
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
