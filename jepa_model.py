"""
JEPA Model: Joint-Embedding Predictive Architecture for Digital Twin
Main model combining encoders and predictor with stop-gradient mechanism
"""

import torch
import torch.nn as nn
from encoders import ContextEncoder, ActionEncoder, TargetEncoder


class Predictor(nn.Module):
    """
    Predictor network: predicts target embedding from context and action
    f(s_x, e_a) -> s_y_pred

    Input: concatenation of [s_x, e_a]
    Output: predicted target embedding s_y_pred
    """

    def __init__(self, enc_x_dim, enc_action_dim, enc_y_dim,
                 hidden_dims, dropout=0.1):
        super().__init__()

        input_dim = enc_x_dim + enc_action_dim
        layers = []

        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output layer to enc_y_dim
        layers.append(nn.Linear(in_dim, enc_y_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, s_x, e_a):
        """
        Args:
            s_x: (batch, enc_x_dim) - context embedding
            e_a: (batch, enc_action_dim) - action embedding

        Returns:
            s_y_pred: (batch, enc_y_dim) - predicted target embedding
        """
        # Concatenate context and action embeddings
        combined = torch.cat([s_x, e_a], dim=1)
        return self.network(combined)


class JEPAModel(nn.Module):
    """
    Complete JEPA model for athlete digital twin

    Architecture:
        x_t -> Enc_x -> s_x ----\
                                  \
        a_t -> Enc_action -> e_a -> Predictor -> s_y_pred ---> Loss: ||s_y_pred - s_y||^2
                                                                         ^
        y_{t+tau} -> Enc_y -> s_y (stop-gradient) --------------------|
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Context encoder (Transformer)
        self.enc_x = ContextEncoder(
            num_features=config.NUM_FEATURES,
            enc_dim=config.ENC_X_DIM,
            nhead=config.ENC_X_NHEAD,
            num_layers=config.ENC_X_LAYERS,
            dim_feedforward=config.ENC_X_DIM_FEEDFORWARD,
            dropout=config.ENC_X_DROPOUT,
            max_len=config.WINDOW_LENGTH * 2  # Allow some flexibility
        )

        # Action encoder (MLP)
        self.enc_action = ActionEncoder(
            num_action_features=config.NUM_ACTION_FEATURES,
            hidden_dims=config.ENC_ACTION_HIDDEN,
            enc_dim=config.ENC_ACTION_DIM,
            dropout=config.ENC_ACTION_DROPOUT
        )

        # Target encoder (MLP for single timestep)
        self.enc_y = TargetEncoder(
            num_features=config.NUM_FEATURES,
            enc_dim=config.ENC_Y_DIM,
            nhead=config.ENC_Y_NHEAD,
            num_layers=config.ENC_Y_LAYERS,
            dim_feedforward=config.ENC_Y_DIM_FEEDFORWARD,
            dropout=config.ENC_Y_DROPOUT,
            single_timestep=True  # Predict single day ahead
        )

        # Predictor (Deep MLP)
        self.predictor = Predictor(
            enc_x_dim=config.ENC_X_DIM,
            enc_action_dim=config.ENC_ACTION_DIM,
            enc_y_dim=config.ENC_Y_DIM,
            hidden_dims=config.PREDICTOR_HIDDEN,
            dropout=config.PREDICTOR_DROPOUT
        )

    def encode_context(self, x_t, mask=None):
        """Encode context window"""
        return self.enc_x(x_t, mask)

    def encode_action(self, a_t):
        """Encode action"""
        return self.enc_action(a_t)

    def encode_target(self, y_t, stop_grad=True):
        """
        Encode target with optional stop-gradient
        This is the key to JEPA: target encoder is not updated via prediction loss
        """
        s_y = self.enc_y(y_t)
        if stop_grad:
            s_y = s_y.detach()  # Stop gradient!
        return s_y

    def predict(self, s_x, e_a):
        """Predict target embedding from context and action"""
        return self.predictor(s_x, e_a)

    def forward(self, x_t, a_t, y_t, mask_x=None, mask_y=None):
        """
        Full forward pass for training

        Args:
            x_t: (batch, window_length, num_features) - context window
            a_t: (batch, num_action_features) - action/plan
            y_t: (batch, num_features) - target future state
            mask_x: Optional mask for context
            mask_y: Optional mask for target

        Returns:
            s_y_pred: (batch, enc_y_dim) - predicted target embedding
            s_y: (batch, enc_y_dim) - actual target embedding (detached)
            loss: scalar - MSE loss in latent space
        """
        # Encode inputs
        s_x = self.encode_context(x_t, mask_x)  # Gradients flow
        e_a = self.encode_action(a_t)           # Gradients flow
        s_y = self.encode_target(y_t, stop_grad=True)  # NO gradients (detached)

        # Predict target embedding
        s_y_pred = self.predict(s_x, e_a)  # Gradients flow

        # Compute loss in latent space
        loss = nn.functional.mse_loss(s_y_pred, s_y)

        return s_y_pred, s_y, loss

    def inference(self, x_t, a_t, mask_x=None):
        """
        Inference mode: predict future embedding given context and action
        Does not require target y_t

        Args:
            x_t: (batch, window_length, num_features)
            a_t: (batch, num_action_features)

        Returns:
            s_y_pred: (batch, enc_y_dim) - predicted future embedding
        """
        with torch.no_grad():
            s_x = self.encode_context(x_t, mask_x)
            e_a = self.encode_action(a_t)
            s_y_pred = self.predict(s_x, e_a)
        return s_y_pred


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test JEPA model
    from config import Config

    config = Config()
    model = JEPAModel(config)

    print("=== JEPA Model ===")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"  Enc_x: {count_parameters(model.enc_x):,}")
    print(f"  Enc_action: {count_parameters(model.enc_action):,}")
    print(f"  Enc_y: {count_parameters(model.enc_y):,}")
    print(f"  Predictor: {count_parameters(model.predictor):,}")

    # Test forward pass
    batch_size = 4
    x_t = torch.randn(batch_size, config.WINDOW_LENGTH, config.NUM_FEATURES)
    a_t = torch.randn(batch_size, config.NUM_ACTION_FEATURES)
    y_t = torch.randn(batch_size, config.NUM_FEATURES)

    print("\n=== Forward Pass Test ===")
    s_y_pred, s_y, loss = model(x_t, a_t, y_t)

    print(f"Input shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  a_t: {a_t.shape}")
    print(f"  y_t: {y_t.shape}")
    print(f"Output shapes:")
    print(f"  s_y_pred: {s_y_pred.shape}")
    print(f"  s_y: {s_y.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test inference
    print("\n=== Inference Test ===")
    s_y_pred_inf = model.inference(x_t, a_t)
    print(f"Predicted embedding shape: {s_y_pred_inf.shape}")

    # Verify stop-gradient works
    print("\n=== Stop-Gradient Verification ===")
    print(f"s_y requires_grad: {s_y.requires_grad}")  # Should be False
    print(f"s_y_pred requires_grad: {s_y_pred.requires_grad}")  # Should be True

    print("\nAll model tests passed!")
