#!/usr/bin/env python3
"""
Generate Advanced Visualizations for Shrine Bowl Analytics

Creates:
1. SHAP analysis for individual players
2. Position-specific leaderboards
3. Case study visualizations for sleeper picks
"""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from src.data.pipeline import DataPipeline
from src.models.enhanced_ptp_model import train_enhanced_ptp_model, EnhancedPTPModel
from src.viz.advanced_charts import (
    create_shap_waterfall_chart,
    create_position_leaderboard,
    create_case_study_visualization,
    create_all_position_leaderboards,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP library not available, using approximated feature contributions")


def compute_feature_contributions(
    model: EnhancedPTPModel,
    X: np.ndarray,
    positions: list,
    feature_names: list,
    player_idx: int
) -> dict:
    """
    Compute feature contributions for a single player.
    
    Uses SHAP if available, otherwise approximates with feature importance * normalized value.
    """
    pos = positions[player_idx]
    group = model.get_position_group(pos)
    model_key = group if group in model.models else "global"
    
    if model_key not in model.models:
        return {}
    
    selected = model.selected_features[model_key]
    X_i = X[player_idx:player_idx+1, selected['indices']]
    X_imputed = model.imputers[model_key].transform(X_i)
    X_scaled = model.scalers[model_key].transform(X_imputed)
    
    classifier = model.models[model_key]['classifier']
    
    contributions = {}
    shap_worked = False
    
    if SHAP_AVAILABLE:
        try:
            # Use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_scaled)
            
            for i, feat_name in enumerate(selected['names']):
                contributions[feat_name] = float(shap_values[0, i]) * 10  # Scale for display
            shap_worked = True
        except Exception as e:
            print(f"    SHAP failed: {e}, using approximation")
    
    if not shap_worked:
        # Approximate: importance * normalized deviation from mean
        importances = classifier.feature_importances_
        for i, feat_name in enumerate(selected['names']):
            val = X_scaled[0, i]  # Already z-scored
            contribution = float(importances[i] * val * 15)  # Scale for display
            contributions[feat_name] = contribution
    
    return contributions


def get_feature_values(
    X: np.ndarray,
    feature_names: list,
    player_idx: int
) -> dict:
    """Get raw feature values for a player."""
    values = {}
    for i, name in enumerate(feature_names):
        val = X[player_idx, i]
        if not np.isnan(val):
            values[name] = float(val)
    return values


def find_sleeper_picks(leaderboard: pl.DataFrame, n: int = 5) -> pl.DataFrame:
    """
    Find the best sleeper picks - players with high PTP but late/no draft pick.
    """
    # Sleepers are late round (5-7) or undrafted with high PTP
    sleepers = leaderboard.filter(
        (pl.col('draft_round').is_null() | (pl.col('draft_round') >= 5)) &
        (pl.col('ptp_score') >= 65)
    ).sort('ptp_score', descending=True).head(n)
    
    return sleepers


def main():
    """Generate all advanced visualizations."""
    
    print("=" * 70)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("outputs/enhanced/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load leaderboard
    leaderboard = pl.read_csv("outputs/enhanced/leaderboard.csv")
    
    print("\n[1/4] Creating position-specific leaderboards...")
    create_all_position_leaderboards(leaderboard, output_dir)
    
    # Now train model to get SHAP contributions
    print("\n[2/4] Training model for SHAP analysis...")
    pipeline = DataPipeline()
    
    # Quick training without hyperparameter tuning for speed
    model, _, _ = train_enhanced_ptp_model(
        pipeline,
        include_tracking=True,
        tune_hyperparams=False  # Skip tuning for speed
    )
    
    # Get feature matrix for SHAP
    from src.features.physical_features import extract_physical_features_for_modeling
    from src.features.production_features import extract_production_features_for_modeling
    from src.models.enhanced_ptp_model import extract_tracking_features_batch
    
    outcomes = pipeline.get_player_outcomes()
    player_ids = outcomes.select("player_id").to_series().to_list()
    positions = outcomes.select("position").to_series().to_list()
    
    print("\n  Extracting features for SHAP...")
    physical = extract_physical_features_for_modeling(pipeline, player_ids)
    production = extract_production_features_for_modeling(pipeline, player_ids)
    tracking = extract_tracking_features_batch(pipeline, player_ids)
    
    merged = outcomes.select(["player_id", "total_snaps", "position"])
    merged = merged.join(physical, on="player_id", how="left")
    merged = merged.join(production, on="player_id", how="left")
    merged = merged.join(tracking, on="player_id", how="left")
    
    exclude_cols = ["player_id", "total_snaps", "position", "position_group", "college_position"]
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    numeric_cols = [c for c in feature_cols 
                   if merged.schema.get(c) in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    X = merged.select(numeric_cols).to_numpy()
    
    # Create player_id to index mapping
    player_id_to_idx = {str(pid): i for i, pid in enumerate(player_ids)}
    
    # Find sleeper picks
    print("\n[3/4] Identifying sleeper picks...")
    sleepers = find_sleeper_picks(leaderboard)
    print(f"  Found {len(sleepers)} sleeper candidates")
    
    if len(sleepers) > 0:
        print("\n  Top Sleepers:")
        for row in sleepers.iter_rows(named=True):
            name = row.get('football_name') or f"{row.get('first_name', '')} {row.get('last_name', '')}"
            draft = f"Rd {int(row['draft_round'])}" if row.get('draft_round') else "UDFA"
            print(f"    - {name} ({row['position']}): PTP={row['ptp_score']:.1f}, {draft}, {row['total_snaps']} snaps")
    
    # Generate SHAP and case study for top sleeper
    print("\n[4/4] Creating SHAP and case study visualizations...")
    
    # Generate case studies for top 3 sleepers
    case_study_count = 0
    for i, row in enumerate(sleepers.head(3).iter_rows(named=True)):
        player_id = str(row['player_id'])
        
        if player_id not in player_id_to_idx:
            print(f"  Warning: Player ID {player_id} not found in index")
            continue
        
        player_idx = player_id_to_idx[player_id]
        name = row.get('football_name') or f"{row.get('first_name', '')} {row.get('last_name', '')}"
        position = row['position']
        ptp_score = row['ptp_score']
        draft_round = row.get('draft_round')
        actual_snaps = row['total_snaps']
        predicted_snaps = row['predicted_snaps']
        
        print(f"\n  Creating case study for {name}...")
        
        # Get feature contributions
        contributions = compute_feature_contributions(
            model, X, positions, numeric_cols, player_idx
        )
        
        if len(contributions) == 0:
            print(f"    Warning: No feature contributions for {name}")
            continue
        
        # Get raw feature values
        feature_values = get_feature_values(X, numeric_cols, player_idx)
        
        # Safe filename
        safe_name = name.lower().replace(' ', '_').replace("'", "")
        
        # Create SHAP waterfall
        fig = create_shap_waterfall_chart(
            player_name=name,
            position=position,
            ptp_score=ptp_score,
            feature_contributions=contributions,
            save_path=output_dir / f"shap_{safe_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: shap_{safe_name}.png")
        
        # Create case study
        fig = create_case_study_visualization(
            player_name=name,
            position=position,
            ptp_score=ptp_score,
            draft_round=int(draft_round) if draft_round else None,
            actual_snaps=actual_snaps,
            predicted_snaps=predicted_snaps,
            feature_values=feature_values,
            feature_contributions=contributions,
            save_path=output_dir / f"case_study_{safe_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: case_study_{safe_name}.png")
        case_study_count += 1
    
    # Also create case study for a top performer for comparison
    top_player = leaderboard.head(1).row(0, named=True)
    top_id = str(top_player['player_id'])
    
    if top_id in player_id_to_idx:
        top_idx = player_id_to_idx[top_id]
        top_name = top_player.get('football_name') or f"{top_player.get('first_name', '')} {top_player.get('last_name', '')}"
        
        print(f"\n  Creating case study for top performer: {top_name}...")
        
        contributions = compute_feature_contributions(
            model, X, positions, numeric_cols, top_idx
        )
        feature_values = get_feature_values(X, numeric_cols, top_idx)
        
        safe_name = top_name.lower().replace(' ', '_').replace("'", "")
        
        fig = create_case_study_visualization(
            player_name=top_name,
            position=top_player['position'],
            ptp_score=top_player['ptp_score'],
            draft_round=int(top_player['draft_round']) if top_player.get('draft_round') else None,
            actual_snaps=top_player['total_snaps'],
            predicted_snaps=top_player['predicted_snaps'],
            feature_values=feature_values,
            feature_contributions=contributions,
            save_path=output_dir / f"case_study_{safe_name}.png"
        )
        plt.close(fig)
        print(f"    Saved: case_study_{safe_name}.png")
        case_study_count += 1
    
    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"\nGenerated {case_study_count} case studies")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
