# PTP Score: Predicting Pro Transition Potential

**Shrine Bowl x SumerSports 2026 Analytics Competition**

## Overview

PTP (Pro Transition Potential) Score is a machine learning model that predicts NFL rookie playing time using Shrine Bowl practice tracking data, combine metrics, and college production statistics.

## Key Results

- **113 players analyzed** across 2022-2024 Shrine Bowl cohorts
- **0.65 AUC** for skill position predictions (WR/RB/TE)
- **43M+ tracking records** processed into kinematic features
- Successfully identified late-round stars (Purdy, Pacheco) as high-value before their NFL success

## Repository Structure

```
shrine-bowl-competition/
├── src/
│   ├── data/
│   │   └── pipeline.py          # Data loading and linkage
│   ├── features/
│   │   ├── tracking_features.py # Kinematic feature extraction
│   │   ├── physical_features.py # Combine metric processing
│   │   └── production_features.py # College stats aggregation
│   ├── models/
│   │   ├── ptp_model.py         # Baseline XGBoost model
│   │   └── enhanced_ptp_model.py # Position-specific models
│   └── viz/
│       ├── charts.py            # Standard visualizations
│       └── advanced_charts.py   # SHAP and case study plots
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── outputs/
│   └── enhanced/
│       ├── charts/              # All visualizations
│       ├── leaderboard.csv      # Player rankings
│       └── model_metrics.json   # Model performance
├── train_ptp_model.py           # Baseline training script
├── train_enhanced_model.py      # Enhanced model training
├── generate_advanced_viz.py     # SHAP and case study generation
├── SLIDE_DECK_PROMPT.md         # Presentation content
└── requirements.txt             # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place Shrine Bowl data in `Shrine Bowl Data/` directory
2. Run the enhanced model:

```bash
python train_enhanced_model.py
```

3. Generate advanced visualizations:

```bash
python generate_advanced_viz.py
```

## Methodology

### Feature Engineering

- **Tracking Features**: Speed percentiles, direction changes, acceleration patterns from Zebra RFID data
- **Athletic Indices**: Normalized combine metrics (40-time, vertical, 3-cone) into composite scores
- **Production Metrics**: Dominator rating, per-season efficiency from college stats

### Modeling

- XGBoost with GridSearchCV hyperparameter optimization
- Position-specific models (SKILL, DB, DL, LB, OL)
- Feature selection via Random Forest importance
- SHAP analysis for interpretability

## Key Findings

1. **Practice tracking data reveals hidden value** - Players with elite kinematic profiles often outperform draft position
2. **Position context matters** - Skill positions show strongest predictive signal
3. **Late-round arbitrage exists** - High PTP + low draft capital = opportunity

## Contact

[Your contact information]

## License

This project was created for the Shrine Bowl x SumerSports 2026 Analytics Competition.

