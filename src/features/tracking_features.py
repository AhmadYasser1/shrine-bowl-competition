"""
Tracking Feature Extraction for Shrine Bowl Analytics

Extracts kinematic and athletic features from Zebra RFID tracking data:
- Speed metrics (max, mean, percentiles)
- Acceleration/burst metrics (peak acceleration, jerk)
- Change of direction efficiency
- Movement consistency

These features form the core of the "tracking-derived athletic signature"
in the PTP Score model.
"""

import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.signal import savgol_filter
from pathlib import Path


class TrackingFeatureExtractor:
    """
    Extract player-level kinematic features from tracking data.
    
    Key metrics:
    - max_speed: Peak velocity (yards/second)
    - burst_score: Peak acceleration at movement initiation
    - cod_efficiency: Speed maintained through direction changes
    - consistency: Variation in performance across reps
    """
    
    # Tracking data sample rate (Hz)
    SAMPLE_RATE = 10  # Approximately 10 samples per second
    
    # Smoothing parameters for derivative calculations
    SAVGOL_WINDOW = 5  # Window size for Savitzky-Golay filter
    SAVGOL_ORDER = 2   # Polynomial order
    
    def __init__(self, min_speed_threshold: float = 1.0):
        """
        Initialize the feature extractor.
        
        Args:
            min_speed_threshold: Minimum speed (yards/sec) to consider as "active"
        """
        self.min_speed_threshold = min_speed_threshold
    
    def extract_player_features(
        self,
        tracking_df: pl.DataFrame,
        player_id: str
    ) -> Dict[str, float]:
        """
        Extract all tracking features for a single player.
        
        Args:
            tracking_df: Tracking data DataFrame
            player_id: Player ID to extract features for
            
        Returns:
            Dictionary of feature name -> value
        """
        # Filter to player's data
        player_data = tracking_df.filter(
            pl.col("player_id") == player_id
        ).sort("ts")
        
        if len(player_data) < 10:
            return self._empty_features()
        
        # Extract raw arrays
        speed = player_data.select("s").to_series().to_numpy()
        accel = player_data.select("a").to_series().to_numpy()
        direction = player_data.select("dir").to_series().to_numpy()
        x = player_data.select("x").to_series().to_numpy()
        y = player_data.select("y").to_series().to_numpy()
        
        # Handle NaN values
        speed = np.nan_to_num(speed, nan=0.0)
        accel = np.nan_to_num(accel, nan=0.0)
        direction = np.nan_to_num(direction, nan=0.0)
        
        features = {}
        
        # Speed metrics
        features.update(self._extract_speed_features(speed))
        
        # Acceleration/burst metrics
        features.update(self._extract_burst_features(speed, accel))
        
        # Change of direction metrics
        features.update(self._extract_cod_features(speed, direction, x, y))
        
        # Consistency metrics
        features.update(self._extract_consistency_features(speed, accel))
        
        return features
    
    def _extract_speed_features(self, speed: np.ndarray) -> Dict[str, float]:
        """Extract speed-related features."""
        # Filter to active periods (above threshold)
        active_speed = speed[speed >= self.min_speed_threshold]
        
        if len(active_speed) == 0:
            return {
                "max_speed": 0.0,
                "mean_speed": 0.0,
                "p90_speed": 0.0,
                "p75_speed": 0.0,
                "speed_range": 0.0,
            }
        
        return {
            "max_speed": float(np.max(active_speed)),
            "mean_speed": float(np.mean(active_speed)),
            "p90_speed": float(np.percentile(active_speed, 90)),
            "p75_speed": float(np.percentile(active_speed, 75)),
            "speed_range": float(np.max(active_speed) - np.min(active_speed)),
        }
    
    def _extract_burst_features(
        self,
        speed: np.ndarray,
        accel: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract acceleration and burst features.
        
        Burst = ability to accelerate quickly from low speed
        Jerk = rate of change of acceleration (measures "twitch")
        """
        if len(speed) < self.SAVGOL_WINDOW:
            return {
                "max_accel": 0.0,
                "mean_accel": 0.0,
                "burst_score": 0.0,
                "max_jerk": 0.0,
            }
        
        # Use provided acceleration
        max_accel = float(np.max(np.abs(accel)))
        mean_accel = float(np.mean(np.abs(accel)))
        
        # Burst score: max acceleration when starting from low speed
        # Find moments where speed < 2 and acceleration is high
        low_speed_mask = speed < 2.0
        if np.any(low_speed_mask):
            burst_accels = accel[low_speed_mask]
            burst_score = float(np.max(burst_accels)) if len(burst_accels) > 0 else 0.0
        else:
            burst_score = max_accel
        
        # Jerk: derivative of acceleration
        # Use Savitzky-Golay to smooth before differentiating
        try:
            if len(accel) >= self.SAVGOL_WINDOW:
                smoothed_accel = savgol_filter(accel, self.SAVGOL_WINDOW, self.SAVGOL_ORDER)
                jerk = np.diff(smoothed_accel) * self.SAMPLE_RATE
                max_jerk = float(np.max(np.abs(jerk)))
            else:
                max_jerk = 0.0
        except Exception:
            max_jerk = 0.0
        
        return {
            "max_accel": max_accel,
            "mean_accel": mean_accel,
            "burst_score": burst_score,
            "max_jerk": max_jerk,
        }
    
    def _extract_cod_features(
        self,
        speed: np.ndarray,
        direction: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract change of direction features.
        
        COD efficiency = maintaining speed through direction changes
        Curvature = sharpness of path changes
        """
        if len(speed) < 10:
            return {
                "cod_efficiency": 0.0,
                "max_curvature_at_speed": 0.0,
                "direction_changes": 0,
            }
        
        # Detect direction changes (>30 degree change)
        dir_diff = np.abs(np.diff(direction))
        # Handle wraparound (e.g., 350 to 10 degrees)
        dir_diff = np.minimum(dir_diff, 360 - dir_diff)
        
        # Significant direction changes
        sig_changes = dir_diff > 30
        num_changes = int(np.sum(sig_changes))
        
        # COD efficiency: average speed during direction changes
        if num_changes > 0:
            # Pad to match length
            change_speeds = speed[1:][sig_changes]
            cod_efficiency = float(np.mean(change_speeds)) if len(change_speeds) > 0 else 0.0
        else:
            cod_efficiency = 0.0
        
        # Curvature calculation using position data
        # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5
        try:
            if len(x) >= self.SAVGOL_WINDOW:
                dx = np.gradient(x)
                dy = np.gradient(y)
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                
                # Avoid division by zero
                denom = np.power(dx**2 + dy**2, 1.5) + 1e-6
                curvature = np.abs(dx * ddy - dy * ddx) / denom
                
                # Max curvature while maintaining speed
                high_speed_mask = speed >= 3.0  # At least jogging pace
                if np.any(high_speed_mask):
                    max_curv_at_speed = float(np.max(curvature[high_speed_mask]))
                else:
                    max_curv_at_speed = 0.0
            else:
                max_curv_at_speed = 0.0
        except Exception:
            max_curv_at_speed = 0.0
        
        return {
            "cod_efficiency": cod_efficiency,
            "max_curvature_at_speed": max_curv_at_speed,
            "direction_changes": num_changes,
        }
    
    def _extract_consistency_features(
        self,
        speed: np.ndarray,
        accel: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract consistency/variability features.
        
        Low variability in performance metrics suggests repeatable skill.
        """
        if len(speed) < 10:
            return {
                "speed_std": 0.0,
                "accel_std": 0.0,
                "speed_cv": 0.0,  # Coefficient of variation
            }
        
        active_speed = speed[speed >= self.min_speed_threshold]
        
        if len(active_speed) == 0:
            return {
                "speed_std": 0.0,
                "accel_std": 0.0,
                "speed_cv": 0.0,
            }
        
        speed_std = float(np.std(active_speed))
        speed_mean = float(np.mean(active_speed))
        speed_cv = speed_std / speed_mean if speed_mean > 0 else 0.0
        
        return {
            "speed_std": speed_std,
            "accel_std": float(np.std(accel)),
            "speed_cv": speed_cv,
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty features dict for players with insufficient data."""
        return {
            "max_speed": np.nan,
            "mean_speed": np.nan,
            "p90_speed": np.nan,
            "p75_speed": np.nan,
            "speed_range": np.nan,
            "max_accel": np.nan,
            "mean_accel": np.nan,
            "burst_score": np.nan,
            "max_jerk": np.nan,
            "cod_efficiency": np.nan,
            "max_curvature_at_speed": np.nan,
            "direction_changes": np.nan,
            "speed_std": np.nan,
            "accel_std": np.nan,
            "speed_cv": np.nan,
        }
    
    def extract_all_players(
        self,
        tracking_df: pl.DataFrame,
        player_ids: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Extract features for all players in the tracking data.
        
        Args:
            tracking_df: Tracking DataFrame with player_id column
            player_ids: Optional list of specific players to extract
            
        Returns:
            DataFrame with one row per player and feature columns
        """
        if player_ids is None:
            player_ids = tracking_df.select("player_id").unique().to_series().to_list()
        
        results = []
        for pid in player_ids:
            features = self.extract_player_features(tracking_df, pid)
            features["player_id"] = pid
            results.append(features)
        
        return pl.DataFrame(results)
    
    def extract_by_drill(
        self,
        tracking_df: pl.DataFrame,
        player_ids: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Extract features per player per drill type.
        
        Useful for understanding which drills best showcase each player.
        """
        if "drill_type" not in tracking_df.columns:
            raise ValueError("DataFrame must have drill_type column")
        
        drill_types = tracking_df.select("drill_type").unique().to_series().to_list()
        
        if player_ids is None:
            player_ids = tracking_df.select("player_id").unique().to_series().to_list()
        
        results = []
        for drill in drill_types:
            drill_data = tracking_df.filter(pl.col("drill_type") == drill)
            
            for pid in player_ids:
                features = self.extract_player_features(drill_data, pid)
                features["player_id"] = pid
                features["drill_type"] = drill
                results.append(features)
        
        return pl.DataFrame(results)


def extract_tracking_features_for_modeling(
    pipeline,
    analyzable_player_ids: List[str]
) -> pl.DataFrame:
    """
    Convenience function to extract tracking features for modeling.
    
    Args:
        pipeline: DataPipeline instance
        analyzable_player_ids: List of player IDs to extract features for
        
    Returns:
        DataFrame with player_id and tracking features
    """
    # Load 1-on-1 drill data (most relevant for individual evaluation)
    tracking_lf = pipeline.load_one_on_one_drills(lazy=True)
    
    # Filter to analyzable players and collect
    tracking_df = tracking_lf.filter(
        pl.col("player_id").is_in(analyzable_player_ids)
    ).collect()
    
    if len(tracking_df) == 0:
        # Fall back to all practice data if no 1-on-1 data
        tracking_lf = pipeline.load_practice_tracking(lazy=True)
        tracking_df = tracking_lf.filter(
            pl.col("player_id").is_in(analyzable_player_ids)
        ).collect()
    
    # Extract features
    extractor = TrackingFeatureExtractor()
    features = extractor.extract_all_players(tracking_df, analyzable_player_ids)
    
    return features


if __name__ == "__main__":
    # Test the feature extractor
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.pipeline import DataPipeline
    
    pipeline = DataPipeline()
    
    # Get analyzable players
    analyzable = pipeline.get_analyzable_players()
    player_ids = analyzable.select("player_id").to_series().to_list()[:5]  # Test with 5 players
    
    print(f"Testing feature extraction for {len(player_ids)} players...")
    
    # Load sample tracking data
    tracking = pipeline.load_practice_tracking(lazy=False)
    tracking = tracking.filter(pl.col("player_id").is_in(player_ids))
    
    print(f"Tracking data shape: {tracking.shape}")
    
    # Extract features
    extractor = TrackingFeatureExtractor()
    features = extractor.extract_all_players(tracking, player_ids)
    
    print("\nExtracted features:")
    print(features)


