"""
Visualization Charts for Shrine Bowl Analytics Competition

Creates presentation-ready visualizations:
- Leaderboard chart (top players by PTP Score)
- Sleeper finder chart (PTP Score vs Draft Projection)
- Validation chart (historical predictions vs outcomes)
- Feature importance chart
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple
from pathlib import Path


# Color palette inspired by football field aesthetics
COLORS = {
    "primary": "#1a472a",      # Dark green (field)
    "secondary": "#2d5016",    # Medium green
    "accent": "#c5a100",       # Gold
    "highlight": "#e63946",    # Red for emphasis
    "text": "#1d1d1d",         # Dark text
    "light": "#f8f9fa",        # Light background
    "grid": "#dee2e6",         # Grid lines
}


def set_chart_style():
    """Set consistent chart styling."""
    plt.rcParams.update({
        'figure.facecolor': COLORS["light"],
        'axes.facecolor': COLORS["light"],
        'axes.edgecolor': COLORS["text"],
        'axes.labelcolor': COLORS["text"],
        'text.color': COLORS["text"],
        'xtick.color': COLORS["text"],
        'ytick.color': COLORS["text"],
        'grid.color': COLORS["grid"],
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def create_leaderboard_chart(
    leaderboard: pl.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None,
    title: str = "Top Players by PTP Score"
) -> plt.Figure:
    """
    Create a horizontal bar chart showing top players by PTP Score.
    
    Args:
        leaderboard: DataFrame with player info and ptp_score
        top_n: Number of players to show
        save_path: Optional path to save the figure
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    set_chart_style()
    
    # Get top N players
    top_players = leaderboard.head(top_n)
    
    # Extract data
    names = top_players.select("football_name").to_series().to_list()
    positions = top_players.select("position").to_series().to_list()
    scores = top_players.select("ptp_score").to_series().to_numpy()
    actual_snaps = top_players.select("total_snaps").to_series().to_numpy()
    
    # Create labels with position
    labels = [f"{name} ({pos})" for name, pos in zip(names, positions)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(labels))
    
    # Create horizontal bars
    bars = ax.barh(y_pos, scores, color=COLORS["primary"], alpha=0.8, edgecolor=COLORS["text"])
    
    # Add score labels on bars
    for i, (bar, score, snaps) in enumerate(zip(bars, scores, actual_snaps)):
        width = bar.get_width()
        ax.text(
            width - 3, bar.get_y() + bar.get_height()/2,
            f"{score:.0f}",
            ha='right', va='center', color='white', fontweight='bold', fontsize=11
        )
        # Add actual snaps in parentheses
        ax.text(
            width + 1, bar.get_y() + bar.get_height()/2,
            f"({snaps:.0f} snaps)",
            ha='left', va='center', color=COLORS["text"], fontsize=9
        )
    
    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top player at top
    ax.set_xlabel("PTP Score", fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(0, 110)
    
    # Add subtle grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_sleeper_chart(
    leaderboard: pl.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Sleeper Finder: PTP Score vs Draft Position"
) -> plt.Figure:
    """
    Create a scatter plot identifying potential "sleepers".
    
    Sleepers are players with high PTP Score but low draft position.
    
    Args:
        leaderboard: DataFrame with ptp_score and draft_round
        save_path: Optional path to save the figure
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    set_chart_style()
    
    # Filter to drafted players
    drafted = leaderboard.filter(pl.col("draft_round").is_not_null())
    
    if len(drafted) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No draft data available", ha='center', va='center')
        return fig
    
    # Extract data
    ptp_scores = drafted.select("ptp_score").to_series().to_numpy()
    
    # Convert draft_round to numeric
    draft_rounds = drafted.select(
        pl.col("draft_round").cast(pl.Float64, strict=False)
    ).to_series().to_numpy()
    
    names = drafted.select("football_name").to_series().to_list()
    positions = drafted.select("position").to_series().to_list()
    actual_snaps = drafted.select("total_snaps").to_series().to_numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Color by actual success (snaps)
    scatter = ax.scatter(
        draft_rounds, ptp_scores,
        c=actual_snaps, cmap='RdYlGn',
        s=100, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Actual Rookie Snaps")
    
    # Highlight sleepers (high PTP, late draft)
    median_ptp = np.median(ptp_scores)
    sleeper_mask = (ptp_scores > median_ptp) & (draft_rounds >= 5)
    
    if np.any(sleeper_mask):
        sleeper_indices = np.where(sleeper_mask)[0]
        for idx in sleeper_indices[:5]:  # Annotate top 5 sleepers
            ax.annotate(
                f"{names[idx]}\n({positions[idx]})",
                (draft_rounds[idx], ptp_scores[idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS["highlight"]),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
    
    # Add quadrant lines
    ax.axhline(median_ptp, color=COLORS["grid"], linestyle='--', alpha=0.7)
    ax.axvline(4.5, color=COLORS["grid"], linestyle='--', alpha=0.7)
    
    # Add quadrant labels
    ax.text(1.5, 85, "ELITE\nTop Draft, High PTP", ha='center', fontsize=10, alpha=0.6)
    ax.text(6, 85, "SLEEPER\nLate Draft, High PTP", ha='center', fontsize=10, 
            alpha=0.8, fontweight='bold', color=COLORS["highlight"])
    ax.text(1.5, 35, "OVERDRAFT\nTop Draft, Low PTP", ha='center', fontsize=10, alpha=0.6)
    ax.text(6, 35, "EXPECTED\nLate Draft, Low PTP", ha='center', fontsize=10, alpha=0.6)
    
    # Customize
    ax.set_xlabel("Draft Round", fontweight='bold')
    ax.set_ylabel("PTP Score", fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(range(1, 8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_validation_chart(
    leaderboard: pl.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Model Validation: Predicted vs Actual Snaps"
) -> plt.Figure:
    """
    Create a scatter plot comparing predicted snaps to actual snaps.
    
    Args:
        leaderboard: DataFrame with predicted_snaps and total_snaps
        save_path: Optional path to save the figure
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    set_chart_style()
    
    predicted = leaderboard.select("predicted_snaps").to_series().to_numpy()
    actual = leaderboard.select("total_snaps").to_series().to_numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=60, c=COLORS["primary"], edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(actual), np.max(predicted))
    ax.plot([0, max_val], [0, max_val], '--', color=COLORS["highlight"], 
            linewidth=2, label="Perfect Prediction")
    
    # Calculate correlation
    valid_mask = ~(np.isnan(predicted) | np.isnan(actual))
    if np.sum(valid_mask) > 2:
        corr = np.corrcoef(actual[valid_mask], predicted[valid_mask])[0, 1]
        ax.text(
            0.05, 0.95, f"Correlation: {corr:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Customize
    ax.set_xlabel("Actual Rookie Snaps", fontweight='bold')
    ax.set_ylabel("Predicted Snaps", fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_feature_importance_chart(
    importance_df: pl.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
    title: str = "Top Predictive Features"
) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with feature and importance columns
        top_n: Number of features to show
        save_path: Optional path to save
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    set_chart_style()
    
    top_features = importance_df.head(top_n)
    
    features = top_features.select("feature").to_series().to_list()
    importance = top_features.select("importance").to_series().to_numpy()
    
    # Clean up feature names for display
    display_names = []
    for f in features:
        # Convert snake_case to Title Case
        name = f.replace("_", " ").title()
        # Shorten some common terms
        name = name.replace("Career ", "")
        name = name.replace("Athleticism Index Pos Adj", "Athleticism (Pos-Adjusted)")
        display_names.append(name)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(display_names))
    bars = ax.barh(y_pos, importance, color=COLORS["secondary"], alpha=0.8, edgecolor=COLORS["text"])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002, bar.get_y() + bar.get_height()/2,
            f"{width:.3f}",
            ha='left', va='center', fontsize=9
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_all_charts(
    leaderboard: pl.DataFrame,
    importance_df: pl.DataFrame,
    output_dir: str = "outputs/charts"
) -> List[str]:
    """
    Generate all charts and save to output directory.
    
    Args:
        leaderboard: Player leaderboard DataFrame
        importance_df: Feature importance DataFrame
        output_dir: Directory to save charts
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Leaderboard chart
    fig = create_leaderboard_chart(leaderboard, top_n=10)
    path = output_path / "leaderboard.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(path))
    
    # Sleeper chart
    fig = create_sleeper_chart(leaderboard)
    path = output_path / "sleeper_finder.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(path))
    
    # Validation chart
    fig = create_validation_chart(leaderboard)
    path = output_path / "validation.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(path))
    
    # Feature importance chart
    fig = create_feature_importance_chart(importance_df)
    path = output_path / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(str(path))
    
    return saved_files


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.pipeline import DataPipeline
    from src.models.ptp_model import train_ptp_model
    
    print("=== Generating Visualizations ===\n")
    
    pipeline = DataPipeline()
    
    # Train model to get leaderboard
    model, leaderboard, metrics = train_ptp_model(
        pipeline,
        include_tracking=False,
        model_type="xgboost"
    )
    
    importance = model.get_feature_importance()
    
    # Generate charts
    print("\nGenerating charts...")
    saved_files = generate_all_charts(leaderboard, importance)
    
    print("\nSaved charts:")
    for f in saved_files:
        print(f"  {f}")


