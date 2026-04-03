"""
Dataset and DataLoader for JEPA training
Handles sliding windows, normalization, and action generation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


class AthleteDataset(Dataset):
    """
    PyTorch Dataset for athlete training data

    Creates samples of the form:
        (x_t, a_t, y_t+tau)
    where:
        x_t: context window of length L (e.g., 14 days)
        a_t: action taken at time t
        y_t+tau: target state tau days in the future
    """

    def __init__(self, csv_path, config, split='train', train_ratio=0.7, val_ratio=0.15):
        """
        Args:
            csv_path: Path to CSV file with athlete data
            config: Configuration object
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        self.config = config
        self.split = split

        # Load data
        self.df = pd.read_csv(csv_path)

        # Feature columns
        self.feature_cols = (
            config.REST_FEATURES + config.SESSION_FEATURES
        )

        # Split by athlete for proper temporal validation
        athletes = self.df['athlete_id'].unique()
        n_train = int(len(athletes) * train_ratio)
        n_val = int(len(athletes) * val_ratio)

        if split == 'train':
            self.athletes = athletes[:n_train]
        elif split == 'val':
            self.athletes = athletes[n_train:n_train + n_val]
        else:  # test
            self.athletes = athletes[n_train + n_val:]

        # Filter data for this split
        self.df = self.df[self.df['athlete_id'].isin(self.athletes)].reset_index(drop=True)

        # Compute normalization statistics per athlete (only on training data)
        self.scalers = {}
        self._compute_normalization()

        # Create samples (sliding windows)
        self.samples = self._create_samples()

        print(f"{split.upper()} split: {len(self.samples)} samples from {len(self.athletes)} athletes")

    def _compute_normalization(self):
        """Compute per-athlete normalization statistics"""
        for athlete_id in self.df['athlete_id'].unique():
            athlete_data = self.df[self.df['athlete_id'] == athlete_id][self.feature_cols]

            # Fit scaler on this athlete's data
            scaler = StandardScaler()
            scaler.fit(athlete_data.values)

            self.scalers[athlete_id] = scaler

    def _normalize(self, data, athlete_id):
        """Normalize data using athlete-specific scaler"""
        if athlete_id in self.scalers:
            return self.scalers[athlete_id].transform(data)
        else:
            # Fallback: use mean scaler (shouldn't happen in practice)
            return data

    def _create_samples(self):
        """Create sliding window samples"""
        samples = []

        for athlete_id in self.athletes:
            athlete_df = self.df[self.df['athlete_id'] == athlete_id].sort_values('day')

            # Need at least window_length + tau days
            min_length = self.config.WINDOW_LENGTH + self.config.TAU
            if len(athlete_df) < min_length:
                continue

            # Sliding window
            for i in range(len(athlete_df) - min_length + 1):
                # Context window: days [i, i+window_length)
                context_idx = list(range(i, i + self.config.WINDOW_LENGTH))

                # Target: day i + window_length - 1 + tau
                target_idx = i + self.config.WINDOW_LENGTH - 1 + self.config.TAU

                # Create sample
                samples.append({
                    'athlete_id': athlete_id,
                    'context_indices': context_idx,
                    'target_index': target_idx,
                    'athlete_df_start': athlete_df.index[0]  # For indexing into full df
                })

        return samples

    def _generate_action(self, context_window):
        """
        Generate action based on context
        Simple rule-based approach for now

        Action features: [type_rest, type_easy, type_moderate, type_hard,
                         planned_duration, planned_intensity, planned_elevation, 0]
        """
        # Get last few days to assess fatigue
        recent_days = context_window[-3:]  # Last 3 days

        # Check training load
        recent_duration = recent_days[:, 9].sum()  # duration column
        recent_rpe = recent_days[:, 11].mean()     # rpe column

        # Simple heuristic
        if recent_duration > 180:  # High load -> rest or easy
            if np.random.rand() < 0.5:
                action = [1, 0, 0, 0, 0, 0, 0, 0]  # rest
            else:
                action = [0, 1, 0, 0, 40, 0.7, 0, 0]  # easy
        elif recent_rpe > 7:  # High RPE -> easy
            action = [0, 1, 0, 0, 45, 0.75, 0, 0]
        else:
            # Random training
            choice = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            if choice == 0:  # rest
                action = [1, 0, 0, 0, 0, 0, 0, 0]
            elif choice == 1:  # moderate
                action = [0, 0, 1, 0, 60, 0.85, np.random.uniform(0, 100), 0]
            else:  # hard
                action = [0, 0, 0, 1, 50, 0.95, np.random.uniform(0, 200), 0]

        return np.array(action, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            x_t: (window_length, num_features) - context window
            a_t: (num_action_features,) - action
            y_t: (num_features,) - target state
            mask_x: (window_length,) - mask for context (True = masked)
        """
        sample = self.samples[idx]
        athlete_id = sample['athlete_id']

        # Get athlete's data
        athlete_df = self.df[self.df['athlete_id'] == athlete_id]

        # Extract context window
        context_data = athlete_df.iloc[sample['context_indices']][self.feature_cols].values

        # Extract target
        target_data = athlete_df.iloc[[sample['target_index']]][self.feature_cols].values[0]

        # Normalize
        context_data = self._normalize(context_data, athlete_id)
        target_data = self._normalize(target_data.reshape(1, -1), athlete_id)[0]

        # Create mask (mark rest days - where all session features are 0)
        # Session features are indices 4-11 (after rest features 0-3)
        session_features = context_data[:, 4:]
        is_rest_day = (session_features == 0).all(axis=1)
        mask_x = is_rest_day  # True for rest days

        # Generate action
        a_t = self._generate_action(context_data)

        # Convert to tensors
        x_t = torch.tensor(context_data, dtype=torch.float32)
        a_t = torch.tensor(a_t, dtype=torch.float32)
        y_t = torch.tensor(target_data, dtype=torch.float32)
        mask_x = torch.tensor(mask_x, dtype=torch.bool)

        return x_t, a_t, y_t, mask_x


def create_dataloaders(csv_path, config, batch_size=None, num_workers=0):
    """
    Create train, validation, and test dataloaders

    Args:
        csv_path: Path to CSV file
        config: Configuration object
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Create datasets
    train_dataset = AthleteDataset(csv_path, config, split='train')
    val_dataset = AthleteDataset(csv_path, config, split='val')
    test_dataset = AthleteDataset(csv_path, config, split='test')

    # Save scalers from training set for later use
    scalers_path = Path(config.DATA_DIR) / 'scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump(train_dataset.scalers, f)
    print(f"Saved normalization scalers to {scalers_path}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    from config import Config
    import os

    config = Config()

    # Check if data exists
    csv_path = Path(config.DATA_DIR) / 'synthetic_athlete_data.csv'
    if not csv_path.exists():
        print("Generating synthetic data first...")
        from data_generator import DataGenerator
        generator = DataGenerator(config)
        generator.generate_dataset()

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(csv_path, config)

    # Test batch
    print("\n=== Testing Batch ===")
    x_t, a_t, y_t, mask_x = next(iter(train_loader))

    print(f"Batch shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  a_t: {a_t.shape}")
    print(f"  y_t: {y_t.shape}")
    print(f"  mask_x: {mask_x.shape}")

    print(f"\nContext window sample (first sample, last 3 days):")
    print(x_t[0, -3:])

    print(f"\nAction sample (first sample):")
    print(a_t[0])

    print(f"\nTarget sample (first sample):")
    print(y_t[0])

    print(f"\nMask sample (first sample):")
    print(mask_x[0])

    print("\nDataset tests passed!")
