"""
Example usage of the trained JEPA model
Shows how to use the model for predictions
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from config import Config
from jepa_model import JEPAModel


def load_trained_model(model_path, device):
    """Load a trained JEPA model"""
    config = Config()
    model = JEPAModel(config).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def predict_future_state(model, context_window, planned_action, device):
    """
    Predict future state embedding given context and planned action

    Args:
        model: Trained JEPA model
        context_window: (14, 12) array of recent days
        planned_action: (8,) array of action features

    Returns:
        predicted_embedding: (128,) predicted future state in latent space
    """
    # Convert to tensors
    x_t = torch.tensor(context_window, dtype=torch.float32).unsqueeze(0).to(device)
    a_t = torch.tensor(planned_action, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        s_y_pred = model.inference(x_t, a_t)

    return s_y_pred.squeeze(0).cpu().numpy()


def compare_actions(model, context_window, actions_to_compare, device):
    """
    Compare multiple potential actions and their predicted outcomes

    Args:
        model: Trained JEPA model
        context_window: (14, 12) array
        actions_to_compare: list of (name, action_array) tuples

    Returns:
        predictions: dict mapping action name to predicted embedding
    """
    predictions = {}

    for action_name, action in actions_to_compare:
        pred = predict_future_state(model, context_window, action, device)
        predictions[action_name] = pred

    return predictions


def main():
    """Example usage demonstration"""

    print("=" * 70)
    print("JEPA Digital Twin - Example Usage")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('./models/best_model.pt')

    if not model_path.exists():
        print("\nModel not found. Please train first with: python train.py")
        return

    # Load model
    print("\n1. Loading trained model...")
    model, config = load_trained_model(model_path, device)
    print(f"   Model loaded from {model_path}")

    # Load a real context window from the dataset
    print("\n2. Loading example data...")
    csv_path = Path(config.DATA_DIR) / 'synthetic_athlete_data.csv'
    df = pd.read_csv(csv_path)

    # Get athlete 0's last 14 days
    athlete_df = df[df['athlete_id'] == 0].tail(14)
    feature_cols = config.REST_FEATURES + config.SESSION_FEATURES
    context_window = athlete_df[feature_cols].values

    print(f"   Context: Last 14 days of Athlete 0")
    print(f"   Shape: {context_window.shape}")

    # Load normalization scaler
    scalers_path = Path(config.DATA_DIR) / 'scalers.pkl'
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Normalize context
    if 0 in scalers:
        context_window = scalers[0].transform(context_window)
        print(f"   Context normalized using athlete 0's scaler")

    # Define different actions to compare
    print("\n3. Defining actions to compare...")
    actions_to_compare = [
        ("Rest Day", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
        ("Easy Run (45min)", np.array([0, 1, 0, 0, 45, 0.7, 0, 0], dtype=np.float32)),
        ("Moderate Run (60min)", np.array([0, 0, 1, 0, 60, 0.85, 50, 0], dtype=np.float32)),
        ("Hard Session (50min)", np.array([0, 0, 0, 1, 50, 0.95, 100, 0], dtype=np.float32)),
    ]

    for action_name, _ in actions_to_compare:
        print(f"   - {action_name}")

    # Make predictions
    print("\n4. Predicting future states...")
    predictions = compare_actions(model, context_window, actions_to_compare, device)

    # Analyze predictions
    print("\n5. Analyzing predictions...")
    print("   Predicted embedding norms (rough indicator of 'load'):")

    for action_name, pred in predictions.items():
        norm = np.linalg.norm(pred)
        print(f"   {action_name:25s} → Embedding norm: {norm:.4f}")

    # Compare embeddings pairwise
    print("\n6. Comparing predicted outcomes (cosine similarity):")

    action_names = list(predictions.keys())
    for i in range(len(action_names)):
        for j in range(i + 1, len(action_names)):
            name1, name2 = action_names[i], action_names[j]
            pred1, pred2 = predictions[name1], predictions[name2]

            # Cosine similarity
            cos_sim = np.dot(pred1, pred2) / (np.linalg.norm(pred1) * np.linalg.norm(pred2))
            print(f"   {name1:20s} vs {name2:20s} → {cos_sim:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("""
- Higher cosine similarity means similar predicted future states
- Rest vs Easy should have higher similarity than Rest vs Hard
- The model has learned to distinguish different action impacts
- Embeddings can be used for downstream tasks (fatigue, risk, etc.)
    """)

    print("\nExample completed!")
    print("\nNext steps:")
    print("  - Train task-specific heads on top of embeddings")
    print("  - Use embeddings for action optimization")
    print("  - Build athlete dashboards with predictions")


if __name__ == '__main__':
    main()
