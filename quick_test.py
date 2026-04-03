"""
Quick test of the complete training pipeline
Runs a few epochs to verify everything works
"""

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from config import Config
from jepa_model import JEPAModel, count_parameters
from dataset import create_dataloaders


def main():
    # Configuration
    config = Config()
    config.NUM_EPOCHS = 5  # Just 5 epochs for quick test

    # Setup paths
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    Path(config.LOGS_DIR).mkdir(exist_ok=True)

    # Create dataloaders
    csv_path = Path(config.DATA_DIR) / 'synthetic_athlete_data.csv'
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(csv_path, config)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating JEPA model...")
    model = JEPAModel(config).to(device)
    print(f"Total parameters: {count_parameters(model):,}\n")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Quick training test
    print(f"Running {config.NUM_EPOCHS} epochs for testing...")
    print("=" * 70)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []

        for x_t, a_t, y_t, mask_x in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            y_t = y_t.to(device)
            mask_x = mask_x.to(device)

            optimizer.zero_grad()
            s_y_pred, s_y, loss = model(x_t, a_t, y_t, mask_x=mask_x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_t, a_t, y_t, mask_x in val_loader:
                x_t = x_t.to(device)
                a_t = a_t.to(device)
                y_t = y_t.to(device)
                mask_x = mask_x.to(device)

                s_y_pred, s_y, loss = model(x_t, a_t, y_t, mask_x=mask_x)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("\n" + "=" * 70)
    print("Quick test completed successfully!")
    print("\nAll components are working correctly.")
    print("You can now run 'python train.py' for full training.")


if __name__ == '__main__':
    main()
