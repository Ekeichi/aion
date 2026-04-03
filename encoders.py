"""
Encoder networks for JEPA architecture
- Enc_x: Transformer encoder for context window (x_t)
- Enc_action: MLP encoder for action (a_t)
- Enc_y: Transformer encoder for target (y_t+tau) - used with stop-gradient
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    """
    Encoder for context window x_t using Transformer
    Input: (batch, window_length, num_features)
    Output: (batch, enc_dim)
    """

    def __init__(self, num_features, enc_dim, nhead, num_layers,
                 dim_feedforward, dropout=0.1, max_len=100):
        super().__init__()

        self.num_features = num_features
        self.enc_dim = enc_dim

        # Input projection
        self.input_projection = nn.Linear(num_features, enc_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(enc_dim, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # We'll use (seq, batch, features) format
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output: pool over sequence dimension
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, window_length, num_features)
            mask: Optional (batch, window_length) - True for positions to mask

        Returns:
            s_x: (batch, enc_dim) - latent representation
        """
        batch_size, seq_len, _ = x.shape

        # Project to encoder dimension
        x = self.input_projection(x)  # (batch, seq_len, enc_dim)

        # Transpose for transformer: (seq_len, batch, enc_dim)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Prepare mask for transformer if provided
        if mask is not None:
            # Transformer expects mask where True means "ignore this position"
            mask = mask  # (batch, seq_len)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # (seq_len, batch, enc_dim)

        # Pool over sequence: transpose back and pool
        x = x.transpose(0, 1)  # (batch, seq_len, enc_dim)
        x = x.transpose(1, 2)  # (batch, enc_dim, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch, enc_dim)

        return x


class ActionEncoder(nn.Module):
    """
    Encoder for action a_t using MLP
    Input: (batch, num_action_features)
    Output: (batch, enc_action_dim)
    """

    def __init__(self, num_action_features, hidden_dims, enc_dim, dropout=0.1):
        super().__init__()

        layers = []
        in_dim = num_action_features

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, enc_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, a):
        """
        Args:
            a: (batch, num_action_features)

        Returns:
            e_a: (batch, enc_dim)
        """
        return self.network(a)


class TargetEncoder(nn.Module):
    """
    Encoder for target y_{t+tau} - identical to ContextEncoder
    Used with stop-gradient during training
    Can encode either a single future state or a future window
    """

    def __init__(self, num_features, enc_dim, nhead, num_layers,
                 dim_feedforward, dropout=0.1, max_len=100,
                 single_timestep=True):
        super().__init__()

        self.num_features = num_features
        self.enc_dim = enc_dim
        self.single_timestep = single_timestep

        if single_timestep:
            # For single future day: simple MLP
            self.encoder = nn.Sequential(
                nn.Linear(num_features, enc_dim * 2),
                nn.LayerNorm(enc_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(enc_dim * 2, enc_dim),
                nn.LayerNorm(enc_dim),
                nn.GELU(),
                nn.Linear(enc_dim, enc_dim)
            )
        else:
            # For future window: use Transformer like ContextEncoder
            self.input_projection = nn.Linear(num_features, enc_dim)
            self.pos_encoder = PositionalEncoding(enc_dim, max_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=enc_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, y, mask=None):
        """
        Args:
            y: (batch, num_features) if single_timestep
               or (batch, window_length, num_features) if not
            mask: Optional mask for sequence

        Returns:
            s_y: (batch, enc_dim)
        """
        if self.single_timestep:
            return self.encoder(y)
        else:
            # Same as ContextEncoder
            x = self.input_projection(y)
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = self.transformer(x, src_key_padding_mask=mask)
            x = x.transpose(0, 1).transpose(1, 2)
            x = self.pooling(x).squeeze(-1)
            return x


if __name__ == '__main__':
    # Test encoders
    from config import Config
    config = Config()

    batch_size = 4

    # Test ContextEncoder
    print("Testing ContextEncoder...")
    enc_x = ContextEncoder(
        num_features=config.NUM_FEATURES,
        enc_dim=config.ENC_X_DIM,
        nhead=config.ENC_X_NHEAD,
        num_layers=config.ENC_X_LAYERS,
        dim_feedforward=config.ENC_X_DIM_FEEDFORWARD,
        dropout=config.ENC_X_DROPOUT
    )
    x = torch.randn(batch_size, config.WINDOW_LENGTH, config.NUM_FEATURES)
    s_x = enc_x(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {s_x.shape}")
    assert s_x.shape == (batch_size, config.ENC_X_DIM)

    # Test ActionEncoder
    print("\nTesting ActionEncoder...")
    enc_action = ActionEncoder(
        num_action_features=config.NUM_ACTION_FEATURES,
        hidden_dims=config.ENC_ACTION_HIDDEN,
        enc_dim=config.ENC_ACTION_DIM,
        dropout=config.ENC_ACTION_DROPOUT
    )
    a = torch.randn(batch_size, config.NUM_ACTION_FEATURES)
    e_a = enc_action(a)
    print(f"  Input shape: {a.shape}")
    print(f"  Output shape: {e_a.shape}")
    assert e_a.shape == (batch_size, config.ENC_ACTION_DIM)

    # Test TargetEncoder (single timestep)
    print("\nTesting TargetEncoder (single timestep)...")
    enc_y = TargetEncoder(
        num_features=config.NUM_FEATURES,
        enc_dim=config.ENC_Y_DIM,
        nhead=config.ENC_Y_NHEAD,
        num_layers=config.ENC_Y_LAYERS,
        dim_feedforward=config.ENC_Y_DIM_FEEDFORWARD,
        dropout=config.ENC_Y_DROPOUT,
        single_timestep=True
    )
    y = torch.randn(batch_size, config.NUM_FEATURES)
    s_y = enc_y(y)
    print(f"  Input shape: {y.shape}")
    print(f"  Output shape: {s_y.shape}")
    assert s_y.shape == (batch_size, config.ENC_Y_DIM)

    print("\nAll encoder tests passed!")
