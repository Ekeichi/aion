"""
Synthetic data generator for athlete digital twin
Generates realistic training data with coherent patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import Config


class AthleteProfile:
    """Profile for a synthetic athlete with physiological characteristics"""

    def __init__(self, athlete_id, seed=None):
        if seed is not None:
            np.random.seed(seed + athlete_id)

        self.athlete_id = athlete_id

        # Base physiological parameters
        self.hr_max = np.random.uniform(180, 200)
        self.hr_rest_base = np.random.uniform(45, 65)
        self.hrv_base = np.random.uniform(40, 80)
        self.base_pace = np.random.uniform(3.0, 5.0)  # m/s (12-20 min/km)
        self.cadence_base = np.random.uniform(160, 180)

        # Training characteristics
        self.fitness_level = np.random.uniform(0.5, 1.0)  # 0=beginner, 1=elite
        self.recovery_rate = np.random.uniform(0.6, 1.0)  # How fast they recover

        # Weekly pattern (more training on certain days)
        self.weekly_pattern = np.random.rand(7)
        self.weekly_pattern = self.weekly_pattern / self.weekly_pattern.sum()


class DataGenerator:
    """Generate synthetic training data for athletes"""

    def __init__(self, config: Config, seed=42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)

    def generate_athlete_data(self, athlete_id: int, num_days: int):
        """Generate training data for one athlete"""
        profile = AthleteProfile(athlete_id, self.seed)

        data = []
        fatigue = 0.0  # Cumulative fatigue state

        for day in range(num_days):
            # Day of week influences training
            dow = day % 7
            training_probability = profile.weekly_pattern[dow]

            # Decide if training day (rest days ~2-3 per week)
            is_training_day = np.random.rand() < training_probability * 0.7

            # Rest signals (always present)
            # HRV decreases with fatigue
            hrv_rmssd = profile.hrv_base * (1 - 0.3 * fatigue) + np.random.normal(0, 5)
            hrv_rmssd = max(20, hrv_rmssd)  # Physiological minimum

            # HR rest increases with fatigue
            hr_rest = profile.hr_rest_base * (1 + 0.2 * fatigue) + np.random.normal(0, 3)
            hr_rest = np.clip(hr_rest, 40, 80)

            # Sleep quality degrades with fatigue
            sleep_duration = np.clip(np.random.normal(7.5, 1.0) - fatigue, 5, 10)
            sleep_quality = np.clip(5 - fatigue * 2 + np.random.normal(0, 0.5), 1, 5)

            if is_training_day and fatigue < 0.8:  # Don't train when too fatigued
                # Decide session intensity
                intensity = np.random.choice(['easy', 'moderate', 'hard'],
                                            p=[0.6, 0.3, 0.1])

                # Duration based on intensity
                if intensity == 'easy':
                    duration = np.random.uniform(30, 60)
                    intensity_factor = 0.7
                    fatigue_cost = 0.15
                elif intensity == 'moderate':
                    duration = np.random.uniform(45, 90)
                    intensity_factor = 0.85
                    fatigue_cost = 0.35
                else:  # hard
                    duration = np.random.uniform(30, 75)
                    intensity_factor = 0.95
                    fatigue_cost = 0.55

                # Heart rate during session
                hr_mean = profile.hr_rest_base + (profile.hr_max - profile.hr_rest_base) * intensity_factor
                hr_mean += np.random.normal(0, 5)

                # HR drift increases with duration and fatigue
                hr_drift = (duration / 60) * 5 * (1 + fatigue) + np.random.normal(0, 2)
                hr_drift = max(0, hr_drift)

                # Pace degrades with fatigue
                pace_mean = profile.base_pace * profile.fitness_level * (1 - 0.2 * fatigue)
                pace_mean += np.random.normal(0, 0.3)

                # Pace/HR ratio (economy) - worse when fatigued
                pace_hr_ratio = pace_mean / hr_mean * 100 * (1 - 0.1 * fatigue)

                # Cadence
                cadence_mean = profile.cadence_base + np.random.normal(0, 5)

                # Elevation (occasional hilly runs)
                elevation_gain = np.random.choice([0, 0, 0, np.random.uniform(50, 300)])

                # RPE correlates with intensity and fatigue
                rpe_base = {'easy': 3, 'moderate': 6, 'hard': 8}[intensity]
                rpe = rpe_base + fatigue * 2 + np.random.normal(0, 0.5)
                rpe = np.clip(rpe, 1, 10)

                # Increase fatigue
                fatigue = min(1.0, fatigue + fatigue_cost * (1 - profile.recovery_rate))

            else:
                # Rest day - all session features to 0
                hr_mean = 0
                hr_drift = 0
                pace_mean = 0
                pace_hr_ratio = 0
                cadence_mean = 0
                duration = 0
                elevation_gain = 0
                rpe = 0

                # Recovery (fatigue decreases on rest days)
                fatigue = max(0.0, fatigue - 0.25 * profile.recovery_rate)

            # Store day data
            day_data = {
                'athlete_id': athlete_id,
                'day': day,
                'hrv_rmssd': hrv_rmssd,
                'hr_rest': hr_rest,
                'sleep_duration': sleep_duration,
                'sleep_quality': sleep_quality,
                'hr_mean': hr_mean,
                'hr_drift': hr_drift,
                'pace_mean': pace_mean,
                'pace_hr_ratio': pace_hr_ratio,
                'cadence_mean': cadence_mean,
                'duration': duration,
                'elevation_gain': elevation_gain,
                'rpe': rpe,
                'is_training': int(is_training_day and duration > 0),
                'fatigue_state': fatigue  # Hidden state for analysis
            }
            data.append(day_data)

        return pd.DataFrame(data)

    def generate_dataset(self):
        """Generate full dataset for all athletes"""
        all_data = []

        print(f"Generating data for {self.config.NUM_ATHLETES} athletes...")
        for athlete_id in range(self.config.NUM_ATHLETES):
            athlete_data = self.generate_athlete_data(
                athlete_id,
                self.config.NUM_DAYS_PER_ATHLETE
            )
            all_data.append(athlete_data)
            print(f"  Athlete {athlete_id}: {len(athlete_data)} days generated")

        df = pd.concat(all_data, ignore_index=True)

        # Save to CSV
        Path(self.config.DATA_DIR).mkdir(exist_ok=True)
        output_path = Path(self.config.DATA_DIR) / 'synthetic_athlete_data.csv'
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Training days: {df['is_training'].sum()} ({df['is_training'].mean()*100:.1f}%)")

        return df

    def generate_action_for_day(self, current_state, profile: AthleteProfile):
        """
        Generate a plausible action (planned session) based on current state
        This simulates a coaching decision
        """
        fatigue = current_state.get('fatigue_state', 0.5)
        days_since_hard = current_state.get('days_since_hard', 3)

        # Action features: [type_rest, type_easy, type_moderate, type_hard,
        #                   planned_duration, planned_intensity, planned_elevation]

        # Simple rule-based planning
        if fatigue > 0.7:
            # Rest day
            action = [1, 0, 0, 0, 0, 0, 0, 0]
        elif fatigue > 0.4 or days_since_hard < 2:
            # Easy day
            duration = np.random.uniform(30, 60)
            action = [0, 1, 0, 0, duration, 0.7, 0, 0]
        elif days_since_hard >= 3 and np.random.rand() < 0.3:
            # Hard day
            duration = np.random.uniform(40, 75)
            action = [0, 0, 0, 1, duration, 0.95, np.random.uniform(0, 200), 0]
        else:
            # Moderate day
            duration = np.random.uniform(45, 90)
            action = [0, 0, 1, 0, duration, 0.85, np.random.uniform(0, 100), 0]

        return np.array(action, dtype=np.float32)


if __name__ == '__main__':
    config = Config()
    generator = DataGenerator(config)
    df = generator.generate_dataset()

    # Show statistics
    print("\n=== Dataset Statistics ===")
    print(df.describe())
