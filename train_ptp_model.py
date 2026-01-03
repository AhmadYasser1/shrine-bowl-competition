#!/usr/bin/env python3
"""
Shrine Bowl Analytics Competition - PTP Score Model Training

This script trains the Practice-to-Pro Translation Score model and generates
all outputs needed for the competition submission.

Usage:
    python train_ptp_model.py

Outputs:
    - outputs/charts/*.png: Visualization charts
    - outputs/leaderboard.csv: Player rankings by PTP Score
    - outputs/model_metrics.json: Model performance metrics
"""

import json
from pathlib import Path
import polars as pl
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.data.pipeline import DataPipeline
from src.models.ptp_model import train_ptp_model
from src.viz.charts import generate_all_charts


def main():
    """Run the full training pipeline."""
    
    print("=" * 60)
    print("SHRINE BOWL x SUMERSPORTS 2026 ANALYTICS COMPETITION")
    print("Practice-to-Pro Translation Score (PTP Score)")
    print("=" * 60)
    print()
    
    # Create output directories
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)
    
    # Initialize pipeline
    print("[1/4] Loading data...")
    pipeline = DataPipeline()
    
    # Get overview
    analyzable = pipeline.get_analyzable_players()
    print(f"      Players with tracking data AND NFL outcomes: {len(analyzable)}")
    
    # Train model
    print("\n[2/4] Training PTP Score model...")
    model, leaderboard, metrics = train_ptp_model(
        pipeline,
        include_tracking=False,  # Tracking features are computationally expensive
        model_type="xgboost"
    )
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    chart_files = generate_all_charts(leaderboard, importance, str(output_dir / "charts"))
    for f in chart_files:
        print(f"      Saved: {f}")
    
    # Save outputs
    print("\n[4/4] Saving outputs...")
    
    # Leaderboard CSV
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard_export = leaderboard.select([
        "rank", "player_id", "football_name", "first_name", "last_name",
        "position", "ptp_score", "contributor_prob", "predicted_snaps",
        "total_snaps", "draft_round"
    ])
    leaderboard_export.write_csv(leaderboard_path)
    print(f"      Leaderboard: {leaderboard_path}")
    
    # Metrics JSON
    metrics_path = output_dir / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics: {metrics_path}")
    
    # Feature importance CSV
    importance_path = output_dir / "feature_importance.csv"
    importance.write_csv(importance_path)
    print(f"      Feature importance: {importance_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    print("\nModel Performance:")
    print(f"  Classification AUC: {metrics['class_auc_mean']:.3f} (+/- {metrics['class_auc_std']:.3f})")
    print(f"  Regression R2: {metrics['reg_r2_mean']:.3f} (+/- {metrics['reg_r2_std']:.3f})")
    
    print("\nTop 5 Players by PTP Score:")
    top5 = leaderboard.head(5)
    for row in top5.iter_rows(named=True):
        print(f"  {row['rank']}. {row['football_name']} ({row['position']}): "
              f"PTP={row['ptp_score']:.1f}, Actual Snaps={row['total_snaps']}")
    
    print("\nTop 5 Predictive Features:")
    for row in importance.head(5).iter_rows(named=True):
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    print("\nOutputs saved to:", output_dir.absolute())
    print("\nNext steps:")
    print("  1. Review charts in outputs/charts/")
    print("  2. Build 10-slide presentation")
    print("  3. Submit by January 11, 2026 @ 11:59 PM ET")


if __name__ == "__main__":
    main()


