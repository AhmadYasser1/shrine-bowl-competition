"""
Practice-to-Pro Translation Score (PTP Score) Model

This module implements the core predictive model that combines:
1. Tracking-derived athletic features
2. Physical/combine metrics
3. College production data

To predict NFL rookie year success (measured by total snaps).
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available for model interpretation")


class PTPScoreModel:
    """
    Practice-to-Pro Translation Score Model.
    
    Combines multiple feature sources to predict NFL rookie success.
    Provides both classification (significant contributor vs not) and
    regression (total snaps) capabilities.
    """
    
    # Snaps threshold for "significant contributor"
    SIGNIFICANT_SNAPS_THRESHOLD = 200
    
    # Feature groups for analysis
    TRACKING_FEATURES = [
        "max_speed", "mean_speed", "p90_speed", "p75_speed", "speed_range",
        "max_accel", "mean_accel", "burst_score", "max_jerk",
        "cod_efficiency", "max_curvature_at_speed", "direction_changes",
        "speed_std", "accel_std", "speed_cv"
    ]
    
    PHYSICAL_FEATURES = [
        "height", "weight", "forty_yd_dash", "speed_score", "three_cone",
        "standing_broad_jump", "standing_vertical", "bench_reps_of_225",
        "size_speed_score", "explosion_index", "agility_index",
        "athleticism_index", "athleticism_index_pos_adj"
    ]
    
    PRODUCTION_FEATURES = [
        "college_seasons", "career_receiving_yards", "career_rushing_yards",
        "career_defense_total_tackles", "career_defense_sacks",
        "scrimmage_yards_per_season", "tds_per_season", "tackles_per_season",
        "sacks_per_season", "production_trajectory", "dominator_proxy",
        "dominator_per_season"
    ]
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the PTP Score model.
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'logistic'
        """
        self.model_type = model_type
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names = []
        self.shap_values = None
        
    def prepare_features(
        self,
        tracking_features: Optional[pl.DataFrame],
        physical_features: Optional[pl.DataFrame],
        production_features: Optional[pl.DataFrame],
        outcomes: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target variables.
        
        Args:
            tracking_features: Tracking-derived features (can be None)
            physical_features: Physical/combine features
            production_features: College production features (can be None)
            outcomes: DataFrame with player_id and target variables
            
        Returns:
            Tuple of (X, y_class, y_reg, feature_names)
        """
        # Start with outcomes as base
        merged = outcomes.select(["player_id", "total_snaps"])
        
        # Join feature sources
        if physical_features is not None:
            merged = merged.join(physical_features, on="player_id", how="left")
        
        if production_features is not None:
            merged = merged.join(production_features, on="player_id", how="left")
        
        if tracking_features is not None:
            merged = merged.join(tracking_features, on="player_id", how="left")
        
        # Identify feature columns (exclude player_id and target)
        exclude_cols = ["player_id", "total_snaps", "position_group", "college_position"]
        feature_cols = [c for c in merged.columns if c not in exclude_cols]
        
        # Filter to numeric columns only
        numeric_cols = []
        for col in feature_cols:
            dtype = merged.schema.get(col)
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                numeric_cols.append(col)
        
        self.feature_names = numeric_cols
        
        # Convert to numpy
        X = merged.select(numeric_cols).to_numpy()
        y_snaps = merged.select("total_snaps").to_series().to_numpy()
        
        # Create binary target (significant contributor)
        y_class = (y_snaps >= self.SIGNIFICANT_SNAPS_THRESHOLD).astype(int)
        
        # Log-transform snaps for regression
        y_reg = np.log1p(y_snaps)
        
        return X, y_class, y_reg, numeric_cols
    
    def train(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray
    ) -> Dict[str, float]:
        """
        Train both classifier and regressor.
        
        Args:
            X: Feature matrix
            y_class: Binary classification target
            y_reg: Regression target (log snaps)
            
        Returns:
            Dictionary of training metrics
        """
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Initialize models based on type
        if self.model_type == "xgboost":
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
            self.regressor = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:  # logistic/ridge
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
            self.regressor = Ridge(alpha=1.0)
        
        # Train
        self.classifier.fit(X_scaled, y_class)
        self.regressor.fit(X_scaled, y_reg)
        
        # Cross-validation scores
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        class_scores = cross_val_score(
            self.classifier, X_scaled, y_class, cv=cv, scoring="roc_auc"
        )
        reg_scores = cross_val_score(
            self.regressor, X_scaled, y_reg, cv=cv, scoring="r2"
        )
        
        metrics = {
            "class_auc_mean": float(np.mean(class_scores)),
            "class_auc_std": float(np.std(class_scores)),
            "reg_r2_mean": float(np.mean(reg_scores)),
            "reg_r2_std": float(np.std(reg_scores)),
        }
        
        return metrics
    
    def predict_ptp_score(self, X: np.ndarray) -> np.ndarray:
        """
        Generate PTP Scores for players.
        
        The PTP Score combines:
        - Probability of being a significant contributor (classification)
        - Expected snap count (regression)
        
        Into a single 0-100 score.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of PTP scores (0-100)
        """
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Get probability of significant contributor
        proba = self.classifier.predict_proba(X_scaled)[:, 1]
        
        # Get predicted log snaps and convert to snaps
        log_snaps_pred = self.regressor.predict(X_scaled)
        snaps_pred = np.expm1(log_snaps_pred)
        
        # Normalize snaps to 0-1 range (cap at 1000 snaps)
        snaps_normalized = np.clip(snaps_pred / 1000, 0, 1)
        
        # Combine: 60% probability, 40% normalized snaps
        ptp_score = 0.6 * proba + 0.4 * snaps_normalized
        
        # Scale to 0-100
        ptp_score = ptp_score * 100
        
        return ptp_score
    
    def get_feature_importance(self) -> pl.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.classifier, "feature_importances_"):
            importance = self.classifier.feature_importances_
        elif hasattr(self.classifier, "coef_"):
            importance = np.abs(self.classifier.coef_[0])
        else:
            importance = np.zeros(len(self.feature_names))
        
        return pl.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort("importance", descending=True)
    
    def explain_predictions(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate SHAP values for model interpretation.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values array or None if SHAP not available
        """
        if not SHAP_AVAILABLE:
            return None
        
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        if self.model_type == "xgboost":
            explainer = shap.TreeExplainer(self.classifier)
        else:
            explainer = shap.Explainer(self.classifier, X_scaled)
        
        self.shap_values = explainer.shap_values(X_scaled)
        
        return self.shap_values
    
    def create_leaderboard(
        self,
        X: np.ndarray,
        player_info: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Create a leaderboard of players ranked by PTP Score.
        
        Args:
            X: Feature matrix
            player_info: DataFrame with player_id and name info
            
        Returns:
            Leaderboard DataFrame
        """
        ptp_scores = self.predict_ptp_score(X)
        
        # Get prediction details
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        proba = self.classifier.predict_proba(X_scaled)[:, 1]
        log_snaps_pred = self.regressor.predict(X_scaled)
        snaps_pred = np.expm1(log_snaps_pred)
        
        leaderboard = player_info.with_columns([
            pl.Series("ptp_score", ptp_scores),
            pl.Series("contributor_prob", proba),
            pl.Series("predicted_snaps", snaps_pred),
        ]).sort("ptp_score", descending=True)
        
        # Add rank
        leaderboard = leaderboard.with_row_index("rank", offset=1)
        
        return leaderboard


def train_ptp_model(
    pipeline,
    tracking_features: Optional[pl.DataFrame] = None,
    include_tracking: bool = True,
    model_type: str = "xgboost"
) -> Tuple[PTPScoreModel, pl.DataFrame, Dict[str, float]]:
    """
    End-to-end training of the PTP Score model.
    
    Args:
        pipeline: DataPipeline instance
        tracking_features: Pre-computed tracking features (optional)
        include_tracking: Whether to include tracking features
        model_type: Model type to use
        
    Returns:
        Tuple of (trained model, leaderboard, metrics)
    """
    from src.features.physical_features import extract_physical_features_for_modeling
    from src.features.production_features import extract_production_features_for_modeling
    
    # Get analyzable players (those with outcomes)
    outcomes = pipeline.get_player_outcomes()
    player_ids = outcomes.select("player_id").to_series().to_list()
    
    print(f"Training on {len(player_ids)} players with NFL outcomes")
    
    # Extract features
    print("Extracting physical features...")
    physical_features = extract_physical_features_for_modeling(pipeline, player_ids)
    
    print("Extracting production features...")
    production_features = extract_production_features_for_modeling(pipeline, player_ids)
    
    # Note: Tracking features are expensive to compute, so they're optional
    if include_tracking and tracking_features is None:
        print("Note: Tracking features not provided, training without them")
        tracking_features = None
    
    # Initialize model
    model = PTPScoreModel(model_type=model_type)
    
    # Prepare features
    print("Preparing feature matrix...")
    X, y_class, y_reg, feature_names = model.prepare_features(
        tracking_features, physical_features, production_features, outcomes
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Train
    print(f"Training {model_type} model...")
    metrics = model.train(X, y_class, y_reg)
    
    print(f"\nModel Performance:")
    print(f"  Classification AUC: {metrics['class_auc_mean']:.3f} (+/- {metrics['class_auc_std']:.3f})")
    print(f"  Regression R2: {metrics['reg_r2_mean']:.3f} (+/- {metrics['reg_r2_std']:.3f})")
    
    # Create leaderboard
    print("\nCreating leaderboard...")
    
    # Get player names from outcomes
    player_info = outcomes.select([
        "player_id", "position", "total_snaps", "draft_round"
    ])
    
    # Try to get names from players data
    players = pipeline.load_players()
    names = players.select([
        "player_id", "first_name", "last_name", "football_name"
    ])
    player_info = player_info.join(names, on="player_id", how="left")
    
    leaderboard = model.create_leaderboard(X, player_info)
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance()
    for row in importance.head(10).iter_rows(named=True):
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, leaderboard, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.pipeline import DataPipeline
    
    print("=== PTP Score Model Training ===\n")
    
    pipeline = DataPipeline()
    
    # Train without tracking features for speed (can add later)
    model, leaderboard, metrics = train_ptp_model(
        pipeline,
        include_tracking=False,
        model_type="xgboost"
    )
    
    print("\n=== Top 10 Players by PTP Score ===")
    top_10 = leaderboard.head(10).select([
        "rank", "football_name", "position", "ptp_score", 
        "predicted_snaps", "total_snaps", "draft_round"
    ])
    print(top_10)

