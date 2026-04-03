"""
Configuration file for JEPA Digital Twin model
"""

class Config:
    # Data dimensions
    WINDOW_LENGTH = 14  # Number of days in context window
    NUM_FEATURES = 12   # Features per day
    NUM_ACTION_FEATURES = 8  # Features for action encoding

    # Feature names (for reference and data generation)
    REST_FEATURES = ['hrv_rmssd', 'hr_rest', 'sleep_duration', 'sleep_quality']
    SESSION_FEATURES = ['hr_mean', 'hr_drift', 'pace_mean', 'pace_hr_ratio',
                        'cadence_mean', 'duration', 'elevation_gain', 'rpe']

    # Model architecture
    # Encoder x (Transformer 1D)
    ENC_X_DIM = 128  # Latent dimension for context encoding
    ENC_X_NHEAD = 4  # Number of attention heads
    ENC_X_LAYERS = 3  # Number of transformer layers
    ENC_X_DIM_FEEDFORWARD = 256  # Feedforward dimension
    ENC_X_DROPOUT = 0.1

    # Encoder action (MLP)
    ENC_ACTION_HIDDEN = [64, 128]  # Hidden layers for action encoder
    ENC_ACTION_DIM = 64  # Latent dimension for action encoding
    ENC_ACTION_DROPOUT = 0.1

    # Encoder y (same as Encoder x, but with stop-gradient)
    ENC_Y_DIM = 128  # Should match ENC_X_DIM for comparison
    ENC_Y_NHEAD = 4
    ENC_Y_LAYERS = 3
    ENC_Y_DIM_FEEDFORWARD = 256
    ENC_Y_DROPOUT = 0.1

    # Predictor (MLP)
    PREDICTOR_HIDDEN = [256, 256, 128]  # Deep MLP
    PREDICTOR_DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5

    # Prediction horizon
    TAU = 1  # Predict tau days ahead (default: next day)

    # Data generation (synthetic)
    NUM_ATHLETES = 10
    NUM_DAYS_PER_ATHLETE = 180  # 6 months of data

    # Paths
    DATA_DIR = './data'
    MODEL_DIR = './models'
    LOGS_DIR = './logs'
