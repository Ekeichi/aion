# JEPA Digital Twin - Implementation Guide

This implementation provides a complete JEPA (Joint-Embedding Predictive Architecture) model for athlete digital twin prediction.

## Project Structure

```
.
├── config.py                  # Configuration and hyperparameters
├── data_generator.py          # Synthetic data generation
├── dataset.py                 # PyTorch Dataset and DataLoader
├── encoders.py                # Encoder networks (Enc_x, Enc_action, Enc_y)
├── jepa_model.py              # Main JEPA model with stop-gradient
├── train.py                   # Training script
├── inference.py               # Inference and visualization
├── requirements.txt           # Python dependencies
└── data/                      # Generated data directory
    ├── synthetic_athlete_data.csv
    └── scalers.pkl
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python data_generator.py
```

This generates training data for 10 athletes over 180 days each, with realistic patterns:
- Rest vs training days
- Fatigue accumulation and recovery
- Physiological signals (HRV, HR, sleep)
- Session characteristics (pace, duration, RPE)

### 2. Test Individual Components

Test encoders:
```bash
python encoders.py
```

Test JEPA model:
```bash
python jepa_model.py
```

Test dataset:
```bash
python dataset.py
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load or generate synthetic data
- Create train/val/test splits (70%/15%/15%)
- Train for 100 epochs (configurable in config.py)
- Save best model to `models/best_model.pt`
- Generate training curves in `logs/`

### 4. Run Inference

```bash
python inference.py
```

This will:
- Load the trained model
- Evaluate on test set
- Generate embedding visualizations
- Report metrics (MSE loss, cosine similarity)

## Architecture Details

### Model Components

1. **Context Encoder (Enc_x)**: Transformer encoder
   - Input: (batch, 14 days, 12 features)
   - Output: (batch, 128) latent representation
   - Uses positional encoding + multi-head attention

2. **Action Encoder (Enc_action)**: MLP
   - Input: (batch, 8) action features
   - Output: (batch, 64) action embedding
   - Simple feedforward network

3. **Target Encoder (Enc_y)**: MLP (with stop-gradient)
   - Input: (batch, 12) future state features
   - Output: (batch, 128) target embedding
   - **Key**: Gradients are blocked during backprop

4. **Predictor**: Deep MLP
   - Input: concat([s_x, e_a]) = (batch, 192)
   - Output: (batch, 128) predicted target embedding
   - Learns dynamics: "what happens if I do action A in context X?"

### Training Process

```
x_t (context) ──→ Enc_x ──→ s_x ─┐
                                   ├──→ Predictor ──→ ŝ_y ──→ Loss = ||ŝ_y - s_y||²
a_t (action)  ──→ Enc_action ──→ e_a ─┘                              ↑
                                                                      │
y_t (target)  ──→ Enc_y ──→ s_y (detached) ──────────────────────────┘
```

**Stop-gradient**: The gradient does NOT flow back through Enc_y. This forces the model to learn meaningful dynamics rather than trivial solutions.

## Data Format

### Input Features (12 per day)

**Rest signals** (always present):
- `hrv_rmssd`: Heart rate variability (ms)
- `hr_rest`: Resting heart rate (bpm)
- `sleep_duration`: Sleep duration (hours)
- `sleep_quality`: Sleep quality score (1-5)

**Session signals** (0 on rest days):
- `hr_mean`: Mean heart rate during session
- `hr_drift`: HR drift (fatigue indicator)
- `pace_mean`: Mean pace (m/s)
- `pace_hr_ratio`: Running economy proxy
- `cadence_mean`: Cadence (steps/min)
- `duration`: Session duration (minutes)
- `elevation_gain`: Elevation gain (meters)
- `rpe`: Rate of perceived exertion (1-10)

### Action Features (8)

- One-hot encoding: [rest, easy, moderate, hard]
- Planned duration
- Planned intensity
- Planned elevation
- Padding

## Configuration

Edit `config.py` to adjust:

- **Data dimensions**: Window length, number of features
- **Model architecture**: Hidden dimensions, number of layers, attention heads
- **Training**: Batch size, learning rate, epochs
- **Data generation**: Number of athletes, days per athlete

## Performance Metrics

The model is evaluated on:

1. **MSE Loss**: Mean squared error in latent space
   - Measures distance between predicted and actual embeddings

2. **Cosine Similarity**: Alignment between prediction and target
   - 1.0 = perfect alignment, 0.0 = orthogonal

## Using Real Data

To use real athlete data instead of synthetic:

1. Format your data as CSV with columns matching the feature names in `config.py`
2. Ensure data includes:
   - `athlete_id`: Unique athlete identifier
   - `day`: Sequential day number
   - All 12 feature columns
3. Replace the CSV path in `train.py` and `inference.py`

Example CSV format:
```csv
athlete_id,day,hrv_rmssd,hr_rest,sleep_duration,sleep_quality,hr_mean,hr_drift,pace_mean,pace_hr_ratio,cadence_mean,duration,elevation_gain,rpe
0,0,65.3,52.1,7.8,4.2,0,0,0,0,0,0,0,0
0,1,63.8,53.2,7.5,4.0,145.2,3.1,4.2,2.89,172.3,45.0,0,3.5
...
```

## Next Steps

### Potential Enhancements

1. **Downstream tasks**: Add task-specific heads
   - Fatigue prediction
   - Injury risk assessment
   - Performance forecasting

2. **Multi-horizon prediction**: Predict τ ∈ {1, 3, 7} days ahead

3. **Attention visualization**: Analyze what the Transformer focuses on

4. **Transfer learning**: Pre-train on large cohort, fine-tune per athlete

5. **Action optimization**: Use the model to suggest optimal training plans

6. **Uncertainty quantification**: Add ensemble or Bayesian layers

## References

- JEPA paper: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022)
- Original README: `README.md`
- Training pipeline diagram: `jepa_training_pipeline.svg`
