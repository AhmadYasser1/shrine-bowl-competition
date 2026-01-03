#!/usr/bin/env python3
"""
Shrine Bowl Analytics Competition - Enhanced PTP Score Model Training

This script trains the enhanced Practice-to-Pro Translation Score model with:
1. Tracking features from 1-on-1 drill data
2. Position-specific models (skill, DB, DL, LB, OL)
3. Hyperparameter tuning via grid search
4. Feature selection using Random Forest importance

Usage:
    python train_enhanced_model.py

Outputs:
    - outputs/enhanced/charts/*.png: Visualization charts
    - outputs/enhanced/leaderboard.csv: Player rankings by PTP Score
    - outputs/enhanced/model_metrics.json: Model performance metrics
    - outputs/enhanced/feature_importance.csv: Feature importance by position
"""

import json
from pathlib import Path
import polars as pl
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from src.data.pipeline import DataPipeline
from src.models.enhanced_ptp_model import train_enhanced_ptp_model
from src.viz.charts import (
    create_leaderboard_chart,
    create_sleeper_chart, 
    create_validation_chart,
    create_feature_importance_chart
)


def main():
    """Run the full enhanced training pipeline."""
    
    print("=" * 70)
    print("SHRINE BOWL x SUMERSPORTS 2026 ANALYTICS COMPETITION")
    print("Enhanced PTP Score Model")
    print("=" * 70)
    print()
    print("Improvements over baseline:")
    print("  1. Tracking features from 1-on-1 drills (speed, burst, COD)")
    print("  2. Position-specific models (skill, DB, DL, LB, OL)")
    print("  3. Hyperparameter tuning via grid search")
    print("  4. Feature selection using RF importance")
    print()
    
    # Create output directories
    output_dir = Path("outputs/enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Train enhanced model
    model, leaderboard, metrics = train_enhanced_ptp_model(
        pipeline,
        include_tracking=True,
        tune_hyperparams=True
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Leaderboard chart
    fig = create_leaderboard_chart(
        leaderboard, top_n=10,
        title="Top 10 Players by Enhanced PTP Score"
    )
    fig.savefig(output_dir / "charts" / "leaderboard.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/charts/leaderboard.png")
    
    # Sleeper chart
    fig = create_sleeper_chart(
        leaderboard,
        title="Sleeper Finder: Enhanced PTP Score vs Draft Position"
    )
    fig.savefig(output_dir / "charts" / "sleeper_finder.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/charts/sleeper_finder.png")
    
    # Validation chart
    fig = create_validation_chart(
        leaderboard,
        title="Model Validation: Enhanced Predictions vs Actual"
    )
    fig.savefig(output_dir / "charts" / "validation.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/charts/validation.png")
    
    # Feature importance (use global model)
    importance = model.get_all_feature_importance()
    global_importance = importance.filter(pl.col("position_group") == "global")
    if len(global_importance) > 0:
        fig = create_feature_importance_chart(
            global_importance.select(["feature", "importance"]),
            title="Top Predictive Features (Global Model)"
        )
        fig.savefig(output_dir / "charts" / "feature_importance.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/charts/feature_importance.png")
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Leaderboard CSV
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.write_csv(leaderboard_path)
    print(f"  Leaderboard: {leaderboard_path}")
    
    # Metrics JSON
    metrics_path = output_dir / "model_metrics.json"
    # Convert numpy types to Python types for JSON
    metrics_clean = {}
    for group, m in metrics.items():
        if m:
            metrics_clean[group] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in m.items()
            }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    
    # Feature importance CSV
    importance_path = output_dir / "feature_importance.csv"
    importance.write_csv(importance_path)
    print(f"  Feature importance: {importance_path}")
    
    # Best hyperparameters
    params_path = output_dir / "best_hyperparameters.json"
    with open(params_path, 'w') as f:
        json.dump(model.best_params, f, indent=2)
    print(f"  Best params: {params_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    print("\nModel Performance by Position Group:")
    for group, m in metrics.items():
        if m and 'auc_mean' in m:
            auc = m['auc_mean']
            r2 = m['r2_mean']
            n = m['n_samples']
            if not np.isnan(auc):
                print(f"  {group.upper():8s}: AUC={auc:.3f}, R2={r2:.3f} (n={n})")
    
    print("\nTop 10 Players by Enhanced PTP Score:")
    top10 = leaderboard.head(10)
    for row in top10.iter_rows(named=True):
        name = row['football_name'] if row['football_name'] else f"{row['first_name']} {row['last_name']}"
        print(f"  {row['rank']:2d}. {name:20s} ({row['position']:2s}): "
              f"PTP={row['ptp_score']:.1f}, Pred={row['predicted_snaps']:.0f}, "
              f"Actual={row['total_snaps']}")
    
    # Calculate prediction accuracy
    pred = leaderboard.select("predicted_snaps").to_series().to_numpy()
    actual = leaderboard.select("total_snaps").to_series().to_numpy()
    correlation = np.corrcoef(pred, actual)[0, 1]
    mape = np.mean(np.abs(pred - actual) / (actual + 1)) * 100
    
    print(f"\nPrediction Quality:")
    print(f"  Correlation (Predicted vs Actual): {correlation:.3f}")
    print(f"  Mean Absolute Percentage Error: {mape:.1f}%")
    
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review charts in outputs/enhanced/charts/")
    print("  2. Build 10-slide presentation")
    print("  3. Submit by January 11, 2026 @ 11:59 PM ET")


if __name__ == "__main__":
    main()


