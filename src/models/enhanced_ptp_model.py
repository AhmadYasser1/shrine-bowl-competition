"""
Enhanced Practice-to-Pro Translation Score (PTP Score) Model

Improvements over baseline:
1. Tracking features from 1-on-1 drill data
2. Position-specific models (skill vs linemen)
3. Hyperparameter tuning with grid search
4. Feature selection to remove noise
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
import json

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class EnhancedPTPModel:
    """
    Enhanced PTP Score Model with:
    - Tracking features
    - Position-specific modeling
    - Hyperparameter tuning
    - Feature selection
    """
    
    SIGNIFICANT_SNAPS_THRESHOLD = 200
    
    # Position groups
    SKILL_POSITIONS = ["WR", "RB", "TE", "FB", "QB"]
    DB_POSITIONS = ["DC", "DS"]
    LB_POSITIONS = ["IB", "OB"]
    DL_POSITIONS = ["DE", "DT"]
    OL_POSITIONS = ["OT", "OG", "OC"]
    
    def __init__(self):
        self.models = {}  # Position-specific models
        self.scalers = {}
        self.imputers = {}
        self.selected_features = {}
        self.feature_names = []
        self.best_params = {}
        
    def get_position_group(self, position: str) -> str:
        """Map position to position group."""
        if position in self.SKILL_POSITIONS:
            return "skill"
        elif position in self.DB_POSITIONS:
            return "db"
        elif position in self.LB_POSITIONS:
            return "lb"
        elif position in self.DL_POSITIONS:
            return "dl"
        elif position in self.OL_POSITIONS:
            return "ol"
        else:
            return "other"
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_features: int = 20
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Select top features using Random Forest importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            Tuple of (X_selected, selected_feature_names, selected_indices)
        """
        # Handle missing values for feature selection
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_imputed, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top n_features
        indices = np.argsort(importances)[-n_features:]
        selected_names = [feature_names[i] for i in indices]
        
        return X[:, indices], selected_names, indices.tolist()
    
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 3
    ) -> Dict:
        """
        Tune XGBoost hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters
        """
        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Parameter grid (reduced for speed)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
        }
        
        # Initialize model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_scaled, y)
        
        return grid_search.best_params_
    
    def train_position_model(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        feature_names: List[str],
        position_group: str,
        tune: bool = True
    ) -> Dict[str, float]:
        """
        Train a model for a specific position group.
        
        Args:
            X: Feature matrix
            y_class: Binary classification target
            y_reg: Regression target
            feature_names: List of feature names
            position_group: Position group name
            tune: Whether to tune hyperparameters
            
        Returns:
            Performance metrics
        """
        if len(X) < 10:
            print(f"      Insufficient data for {position_group} ({len(X)} samples), skipping")
            return {}
        
        # Feature selection
        print(f"      Selecting features for {position_group}...")
        n_features = min(15, X.shape[1], len(X) // 3)  # Limit features based on sample size
        X_selected, selected_names, selected_indices = self.select_features(
            X, y_class, feature_names, n_features=n_features
        )
        self.selected_features[position_group] = {
            'names': selected_names,
            'indices': selected_indices
        }
        
        # Impute and scale
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_selected)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        self.imputers[position_group] = imputer
        self.scalers[position_group] = scaler
        
        # Hyperparameter tuning
        if tune and len(X) >= 20:
            print(f"      Tuning hyperparameters for {position_group}...")
            best_params = self.tune_hyperparameters(X_selected, y_class)
            self.best_params[position_group] = best_params
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8
            }
            self.best_params[position_group] = best_params
        
        # Train classifier
        classifier = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric="logloss"
        )
        
        # Train regressor
        regressor = xgb.XGBRegressor(
            **best_params,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(X_scaled, y_class)
            regressor.fit(X_scaled, y_reg)
        
        self.models[position_group] = {
            'classifier': classifier,
            'regressor': regressor
        }
        
        # Cross-validation scores
        cv = KFold(n_splits=min(5, len(X) // 2), shuffle=True, random_state=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_scores = cross_val_score(
                classifier, X_scaled, y_class, cv=cv, scoring="roc_auc"
            )
            reg_scores = cross_val_score(
                regressor, X_scaled, y_reg, cv=cv, scoring="r2"
            )
        
        return {
            "auc_mean": float(np.mean(class_scores)),
            "auc_std": float(np.std(class_scores)),
            "r2_mean": float(np.mean(reg_scores)),
            "r2_std": float(np.std(reg_scores)),
            "n_samples": len(X),
            "n_features": len(selected_names),
            "selected_features": selected_names[:5],  # Top 5
        }
    
    def train(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        positions: List[str],
        feature_names: List[str],
        tune: bool = True
    ) -> Dict[str, Dict]:
        """
        Train position-specific models.
        
        Args:
            X: Feature matrix
            y_class: Binary classification target
            y_reg: Regression target (log snaps)
            positions: List of positions for each sample
            feature_names: List of feature names
            tune: Whether to tune hyperparameters
            
        Returns:
            Dictionary of metrics per position group
        """
        self.feature_names = feature_names
        metrics = {}
        
        # Group by position
        position_groups = [self.get_position_group(p) for p in positions]
        unique_groups = list(set(position_groups))
        
        print(f"    Training {len(unique_groups)} position-specific models...")
        
        for group in unique_groups:
            mask = np.array([pg == group for pg in position_groups])
            X_group = X[mask]
            y_class_group = y_class[mask]
            y_reg_group = y_reg[mask]
            
            print(f"\n    Position group: {group.upper()} ({np.sum(mask)} players)")
            
            group_metrics = self.train_position_model(
                X_group, y_class_group, y_reg_group,
                feature_names, group, tune=tune
            )
            
            if group_metrics:
                metrics[group] = group_metrics
                print(f"      AUC: {group_metrics['auc_mean']:.3f} (+/- {group_metrics['auc_std']:.3f})")
                print(f"      R2: {group_metrics['r2_mean']:.3f} (+/- {group_metrics['r2_std']:.3f})")
        
        # Also train a global model as fallback
        print(f"\n    Training GLOBAL model ({len(X)} players)...")
        global_metrics = self.train_position_model(
            X, y_class, y_reg, feature_names, "global", tune=tune
        )
        metrics["global"] = global_metrics
        print(f"      AUC: {global_metrics['auc_mean']:.3f} (+/- {global_metrics['auc_std']:.3f})")
        print(f"      R2: {global_metrics['r2_mean']:.3f} (+/- {global_metrics['r2_std']:.3f})")
        
        return metrics
    
    def predict_ptp_score(
        self,
        X: np.ndarray,
        positions: List[str]
    ) -> np.ndarray:
        """
        Generate PTP Scores using position-specific models.
        
        Args:
            X: Feature matrix
            positions: List of positions
            
        Returns:
            Array of PTP scores (0-100)
        """
        ptp_scores = np.zeros(len(X))
        
        for i, pos in enumerate(positions):
            group = self.get_position_group(pos)
            
            # Use position-specific model if available, else global
            if group in self.models:
                model_key = group
            else:
                model_key = "global"
            
            if model_key not in self.models:
                ptp_scores[i] = 50.0  # Default score
                continue
            
            # Get selected features
            selected = self.selected_features[model_key]
            X_i = X[i:i+1, selected['indices']]
            
            # Impute and scale
            X_imputed = self.imputers[model_key].transform(X_i)
            X_scaled = self.scalers[model_key].transform(X_imputed)
            
            # Get predictions
            classifier = self.models[model_key]['classifier']
            regressor = self.models[model_key]['regressor']
            
            proba = classifier.predict_proba(X_scaled)[0, 1]
            log_snaps_pred = regressor.predict(X_scaled)[0]
            snaps_pred = np.expm1(log_snaps_pred)
            
            # Combine: 60% probability, 40% normalized snaps
            snaps_normalized = np.clip(snaps_pred / 1000, 0, 1)
            ptp_score = 0.6 * proba + 0.4 * snaps_normalized
            ptp_scores[i] = ptp_score * 100
        
        return ptp_scores
    
    def get_all_feature_importance(self) -> pl.DataFrame:
        """Get feature importance across all models."""
        all_importance = []
        
        for group, model_dict in self.models.items():
            classifier = model_dict['classifier']
            selected = self.selected_features[group]
            
            if hasattr(classifier, "feature_importances_"):
                for name, imp in zip(selected['names'], classifier.feature_importances_):
                    all_importance.append({
                        "position_group": group,
                        "feature": name,
                        "importance": float(imp)
                    })
        
        return pl.DataFrame(all_importance).sort("importance", descending=True)


def extract_tracking_features_batch(pipeline, player_ids: List[str]) -> pl.DataFrame:
    """
    Extract tracking features for all players in batches.
    
    This is computationally expensive but provides valuable athletic metrics.
    """
    from src.features.tracking_features import TrackingFeatureExtractor
    
    print("  Loading 1-on-1 drill tracking data...")
    
    # Load tracking data filtered to relevant players and drills
    tracking_lf = pipeline.load_practice_tracking(
        drill_types=pipeline.ONE_ON_ONE_DRILLS + pipeline.TEAM_DRILLS,
        entity_type="player",
        lazy=True
    )
    
    # Filter to analyzable players
    tracking_lf = tracking_lf.filter(
        pl.col("player_id").is_in(player_ids)
    )
    
    # Sample to reduce memory (take every 5th row for speed)
    print("  Sampling tracking data for efficiency...")
    tracking_df = tracking_lf.collect()
    
    if len(tracking_df) == 0:
        print("  No tracking data found for specified players")
        return pl.DataFrame({"player_id": player_ids})
    
    print(f"  Processing {len(tracking_df):,} tracking records for {len(player_ids)} players...")
    
    # Extract features
    extractor = TrackingFeatureExtractor()
    features = extractor.extract_all_players(tracking_df, player_ids)
    
    return features


def train_enhanced_ptp_model(
    pipeline,
    include_tracking: bool = True,
    tune_hyperparams: bool = True
) -> Tuple[EnhancedPTPModel, pl.DataFrame, Dict]:
    """
    Train the enhanced PTP Score model.
    
    Args:
        pipeline: DataPipeline instance
        include_tracking: Whether to include tracking features
        tune_hyperparams: Whether to tune hyperparameters
        
    Returns:
        Tuple of (model, leaderboard, metrics)
    """
    from src.features.physical_features import extract_physical_features_for_modeling
    from src.features.production_features import extract_production_features_for_modeling
    
    # Get analyzable players
    outcomes = pipeline.get_player_outcomes()
    player_ids = outcomes.select("player_id").to_series().to_list()
    positions = outcomes.select("position").to_series().to_list()
    
    print(f"\n[1/5] Loading data for {len(player_ids)} players...")
    
    # Extract features
    print("\n[2/5] Extracting features...")
    
    print("  Physical features...")
    physical_features = extract_physical_features_for_modeling(pipeline, player_ids)
    
    print("  Production features...")
    production_features = extract_production_features_for_modeling(pipeline, player_ids)
    
    tracking_features = None
    if include_tracking:
        print("  Tracking features (this may take a while)...")
        tracking_features = extract_tracking_features_batch(pipeline, player_ids)
        print(f"  Tracking features extracted: {tracking_features.shape}")
    
    # Merge features
    print("\n[3/5] Building feature matrix...")
    
    merged = outcomes.select(["player_id", "total_snaps", "position"])
    merged = merged.join(physical_features, on="player_id", how="left")
    merged = merged.join(production_features, on="player_id", how="left")
    
    if tracking_features is not None and len(tracking_features) > 0:
        merged = merged.join(tracking_features, on="player_id", how="left")
    
    # Get feature columns
    exclude_cols = ["player_id", "total_snaps", "position", "position_group", "college_position"]
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    
    # Filter to numeric columns
    numeric_cols = []
    for col in feature_cols:
        dtype = merged.schema.get(col)
        if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            numeric_cols.append(col)
    
    print(f"  Total features: {len(numeric_cols)}")
    
    # Convert to numpy
    X = merged.select(numeric_cols).to_numpy()
    y_snaps = merged.select("total_snaps").to_series().to_numpy()
    y_class = (y_snaps >= EnhancedPTPModel.SIGNIFICANT_SNAPS_THRESHOLD).astype(int)
    y_reg = np.log1p(y_snaps)
    
    print(f"  Class balance: {np.sum(y_class)}/{len(y_class)} significant contributors")
    
    # Train enhanced model
    print("\n[4/5] Training enhanced model...")
    model = EnhancedPTPModel()
    metrics = model.train(
        X, y_class, y_reg, positions, numeric_cols,
        tune=tune_hyperparams
    )
    
    # Create leaderboard
    print("\n[5/5] Creating leaderboard...")
    ptp_scores = model.predict_ptp_score(X, positions)
    
    # Get player info
    players = pipeline.load_players()
    names = players.select(["player_id", "first_name", "last_name", "football_name"])
    
    player_info = merged.select(["player_id", "position", "total_snaps"]).join(
        names, on="player_id", how="left"
    )
    
    # Get predicted snaps for display
    predicted_snaps = []
    for i, pos in enumerate(positions):
        group = model.get_position_group(pos)
        model_key = group if group in model.models else "global"
        
        if model_key in model.models:
            selected = model.selected_features[model_key]
            X_i = X[i:i+1, selected['indices']]
            X_imputed = model.imputers[model_key].transform(X_i)
            X_scaled = model.scalers[model_key].transform(X_imputed)
            log_snaps = model.models[model_key]['regressor'].predict(X_scaled)[0]
            predicted_snaps.append(float(np.expm1(log_snaps)))
        else:
            predicted_snaps.append(0.0)
    
    leaderboard = player_info.with_columns([
        pl.Series("ptp_score", ptp_scores),
        pl.Series("predicted_snaps", predicted_snaps),
    ]).sort("ptp_score", descending=True)
    
    leaderboard = leaderboard.with_row_index("rank", offset=1)
    
    # Add draft info
    draft_info = outcomes.select(["player_id", "draft_round"])
    leaderboard = leaderboard.join(draft_info, on="player_id", how="left")
    
    return model, leaderboard, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.pipeline import DataPipeline
    
    print("=" * 70)
    print("ENHANCED PTP SCORE MODEL TRAINING")
    print("With tracking features, position-specific models, and hyperparameter tuning")
    print("=" * 70)
    
    pipeline = DataPipeline()
    
    model, leaderboard, metrics = train_enhanced_ptp_model(
        pipeline,
        include_tracking=True,
        tune_hyperparams=True
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    print("\nPosition-Specific Model Performance:")
    for group, m in metrics.items():
        if m:
            print(f"\n  {group.upper()}:")
            print(f"    Samples: {m['n_samples']}, Features: {m['n_features']}")
            print(f"    AUC: {m['auc_mean']:.3f} (+/- {m['auc_std']:.3f})")
            print(f"    R2: {m['r2_mean']:.3f} (+/- {m['r2_std']:.3f})")
            print(f"    Top features: {', '.join(m['selected_features'])}")
    
    print("\n\nTop 10 Players by Enhanced PTP Score:")
    print(leaderboard.head(10).select([
        "rank", "football_name", "position", "ptp_score", 
        "predicted_snaps", "total_snaps"
    ]))


