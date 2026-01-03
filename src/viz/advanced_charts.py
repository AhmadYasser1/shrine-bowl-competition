"""
Advanced Visualization Charts for Shrine Bowl Analytics

Includes:
- SHAP analysis plots
- Position-specific leaderboards
- Case study deep dives
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import polars as pl
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Color scheme - NFL-inspired
COLORS = {
    'primary': '#013369',      # Navy blue
    'secondary': '#D50A0A',    # Red
    'accent': '#FFB612',       # Gold
    'success': '#2E7D32',      # Green
    'warning': '#F57C00',      # Orange
    'neutral': '#5C6B73',      # Gray
    'bg_dark': '#1A1A2E',
    'bg_light': '#F5F5F5',
    'text_light': '#FFFFFF',
    'text_dark': '#1A1A2E',
}

POSITION_COLORS = {
    'WR': '#FF6B6B',
    'RB': '#4ECDC4',
    'TE': '#45B7D1',
    'QB': '#96CEB4',
    'FB': '#88D8B0',
    'DC': '#DDA0DD',
    'DS': '#9370DB',
    'IB': '#F4A460',
    'OB': '#DAA520',
    'DE': '#CD853F',
    'DT': '#8B4513',
    'OT': '#708090',
    'OG': '#778899',
    'OC': '#696969',
}


def create_shap_waterfall_chart(
    player_name: str,
    position: str,
    ptp_score: float,
    feature_contributions: Dict[str, float],
    base_value: float = 50.0,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a SHAP-style waterfall chart showing feature contributions.
    
    Args:
        player_name: Player's name
        position: Player's position
        ptp_score: Final PTP score
        feature_contributions: Dict of feature name -> contribution value
        base_value: Starting base value
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Sort contributions by absolute value
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]  # Top 10 features
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['bg_light'])
    ax.set_facecolor(COLORS['bg_light'])
    
    # Calculate cumulative values
    labels = ['Base Value'] + [f[0] for f in sorted_features] + ['PTP Score']
    values = [base_value]
    cumulative = base_value
    
    for feature, contribution in sorted_features:
        cumulative += contribution
        values.append(cumulative)
    values.append(ptp_score)
    
    # Create waterfall
    n_bars = len(values)
    x_pos = np.arange(n_bars)
    
    for i in range(n_bars):
        if i == 0:  # Base value
            color = COLORS['neutral']
            ax.bar(i, values[i], color=color, edgecolor='white', linewidth=1.5)
        elif i == n_bars - 1:  # Final value
            color = COLORS['primary']
            ax.bar(i, values[i], color=color, edgecolor='white', linewidth=1.5)
        else:
            prev_val = values[i-1] if i > 0 else base_value
            contribution = sorted_features[i-1][1]
            
            if contribution >= 0:
                color = COLORS['success']
                ax.bar(i, contribution, bottom=prev_val, color=color, 
                      edgecolor='white', linewidth=1.5)
            else:
                color = COLORS['secondary']
                ax.bar(i, abs(contribution), bottom=values[i], color=color,
                      edgecolor='white', linewidth=1.5)
            
            # Add connecting lines
            if i < n_bars - 1:
                ax.plot([i - 0.4, i + 0.4], [values[i], values[i]], 
                       color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add value labels
    for i, (label, val) in enumerate(zip(labels, values)):
        if i == 0 or i == n_bars - 1:
            ax.text(i, val + 2, f'{val:.1f}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color=COLORS['text_dark'])
        else:
            contribution = sorted_features[i-1][1]
            sign = '+' if contribution >= 0 else ''
            ax.text(i, val + 1, f'{sign}{contribution:.1f}', ha='center', va='bottom',
                   fontsize=9, color=COLORS['text_dark'])
    
    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('PTP Score', fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP Analysis: {player_name} ({position})\nFeature Contributions to PTP Score',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLORS['success'], label='Positive Impact'),
        mpatches.Patch(color=COLORS['secondary'], label='Negative Impact'),
        mpatches.Patch(color=COLORS['primary'], label='Final Score'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    ax.set_ylim(0, max(values) + 15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg_light'])
    
    return fig


def create_position_leaderboard(
    leaderboard: pl.DataFrame,
    position_group: str,
    positions: List[str],
    title: str,
    save_path: Optional[Path] = None,
    top_n: int = 10
) -> plt.Figure:
    """
    Create a position-specific leaderboard chart.
    
    Args:
        leaderboard: Full leaderboard DataFrame
        position_group: Name of position group (e.g., "Skill", "DB")
        positions: List of positions to include
        title: Chart title
        save_path: Optional path to save
        top_n: Number of players to show
        
    Returns:
        Matplotlib figure
    """
    # Filter to positions
    filtered = leaderboard.filter(pl.col('position').is_in(positions))
    
    if len(filtered) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No players found for {position_group}", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Re-rank within position group
    filtered = filtered.sort('ptp_score', descending=True).head(top_n)
    filtered = filtered.with_row_index('pos_rank', offset=1)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(filtered) * 0.6)), 
                          facecolor=COLORS['bg_light'])
    ax.set_facecolor(COLORS['bg_light'])
    
    # Create horizontal bar chart
    y_pos = np.arange(len(filtered))[::-1]
    scores = filtered.select('ptp_score').to_series().to_numpy()
    
    # Color by position
    colors = [POSITION_COLORS.get(pos, COLORS['neutral']) 
              for pos in filtered.select('position').to_series().to_list()]
    
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=1.5, height=0.7)
    
    # Add player info
    for i, row in enumerate(filtered.iter_rows(named=True)):
        name = row.get('football_name') or f"{row.get('first_name', '')} {row.get('last_name', '')}"
        pos = row['position']
        draft_round = row.get('draft_round')
        actual_snaps = row['total_snaps']
        
        # Draft info
        draft_str = f"Rd {int(draft_round)}" if draft_round else "UDFA"
        
        # Label inside bar
        label = f"{name} ({pos}) - {draft_str}"
        ax.text(2, y_pos[i], label, va='center', ha='left',
               fontsize=10, fontweight='bold', color=COLORS['text_dark'])
        
        # Score at end of bar
        ax.text(scores[i] + 1, y_pos[i], f'{scores[i]:.1f}',
               va='center', ha='left', fontsize=11, fontweight='bold',
               color=COLORS['primary'])
        
        # Actual snaps annotation
        ax.text(scores[i] - 2, y_pos[i], f'{actual_snaps} snaps',
               va='center', ha='right', fontsize=8, color='white', alpha=0.9)
    
    # Styling
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, len(filtered) - 0.5)
    ax.set_yticks([])
    ax.set_xlabel('PTP Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=COLORS['text_dark'])
    
    ax.axvline(x=50, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Average')
    ax.axvline(x=70, color=COLORS['success'], linestyle='--', alpha=0.5, label='High Potential')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Legend for threshold lines
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg_light'])
    
    return fig


def create_case_study_visualization(
    player_name: str,
    position: str,
    ptp_score: float,
    draft_round: Optional[int],
    actual_snaps: int,
    predicted_snaps: float,
    feature_values: Dict[str, float],
    feature_contributions: Dict[str, float],
    college_stats: Optional[Dict] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a comprehensive case study visualization for a player.
    
    Args:
        player_name: Player's name
        position: Position
        ptp_score: PTP score
        draft_round: Draft round (None if UDFA)
        actual_snaps: Actual rookie snaps
        predicted_snaps: Model predicted snaps
        feature_values: Dict of raw feature values
        feature_contributions: Dict of SHAP contributions
        college_stats: Optional college statistics
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['bg_light'])
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title area
    fig.suptitle(f'CASE STUDY: {player_name.upper()}', 
                fontsize=20, fontweight='bold', y=0.98, color=COLORS['primary'])
    
    # 1. Player Card (top left)
    ax_card = fig.add_subplot(gs[0, 0])
    ax_card.set_facecolor(COLORS['primary'])
    ax_card.set_xlim(0, 10)
    ax_card.set_ylim(0, 10)
    
    draft_str = f"Round {draft_round}" if draft_round else "Undrafted"
    
    ax_card.text(5, 8, player_name, ha='center', va='center', 
                fontsize=18, fontweight='bold', color='white')
    ax_card.text(5, 6.5, position, ha='center', va='center',
                fontsize=14, color=COLORS['accent'])
    ax_card.text(5, 5, f'Draft: {draft_str}', ha='center', va='center',
                fontsize=12, color='white')
    ax_card.text(5, 3.5, f'PTP Score: {ptp_score:.1f}', ha='center', va='center',
                fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax_card.text(5, 2, f'Predicted: {predicted_snaps:.0f} snaps', ha='center', va='center',
                fontsize=11, color='white')
    ax_card.text(5, 1, f'Actual: {actual_snaps} snaps', ha='center', va='center',
                fontsize=11, color=COLORS['success'] if actual_snaps > predicted_snaps * 0.8 else COLORS['secondary'])
    
    ax_card.axis('off')
    
    # 2. PTP Score Gauge (top middle)
    ax_gauge = fig.add_subplot(gs[0, 1])
    ax_gauge.set_facecolor(COLORS['bg_light'])
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r_outer = 1
    r_inner = 0.6
    
    # Background arc
    ax_gauge.fill_between(theta, r_inner, r_outer, alpha=0.2, color=COLORS['neutral'])
    
    # Score arc
    score_theta = np.linspace(0, np.pi * (ptp_score / 100), 50)
    color = COLORS['success'] if ptp_score >= 70 else COLORS['warning'] if ptp_score >= 50 else COLORS['secondary']
    ax_gauge.fill_between(score_theta, r_inner, r_outer, alpha=0.8, color=color)
    
    # Center text
    ax_gauge.text(np.pi/2, 0.3, f'{ptp_score:.0f}', ha='center', va='center',
                 fontsize=32, fontweight='bold', color=COLORS['primary'],
                 transform=ax_gauge.transData)
    ax_gauge.text(np.pi/2, -0.1, 'PTP Score', ha='center', va='center',
                 fontsize=12, color=COLORS['text_dark'],
                 transform=ax_gauge.transData)
    
    ax_gauge.set_xlim(-0.2, np.pi + 0.2)
    ax_gauge.set_ylim(-0.3, 1.2)
    ax_gauge.set_aspect('equal')
    ax_gauge.axis('off')
    ax_gauge.set_title('Performance Potential', fontsize=12, fontweight='bold')
    
    # 3. Prediction vs Actual (top right)
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_pred.set_facecolor(COLORS['bg_light'])
    
    categories = ['Predicted', 'Actual']
    values = [predicted_snaps, actual_snaps]
    colors = [COLORS['primary'], COLORS['success']]
    
    bars = ax_pred.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, values):
        ax_pred.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Error percentage
    error = abs(predicted_snaps - actual_snaps) / actual_snaps * 100
    ax_pred.text(0.5, 0.95, f'Error: {error:.1f}%', transform=ax_pred.transAxes,
                ha='center', va='top', fontsize=10, color=COLORS['neutral'])
    
    ax_pred.set_ylabel('Rookie Snaps', fontsize=11)
    ax_pred.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    
    # 4. Feature Contributions (middle row, full width)
    ax_shap = fig.add_subplot(gs[1, :])
    ax_shap.set_facecolor(COLORS['bg_light'])
    
    # Sort and take top 8 features
    sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    
    features = [f[0] for f in sorted_features]
    contributions = [f[1] for f in sorted_features]
    colors = [COLORS['success'] if c > 0 else COLORS['secondary'] for c in contributions]
    
    y_pos = np.arange(len(features))
    ax_shap.barh(y_pos, contributions, color=colors, edgecolor='white', linewidth=1.5)
    
    ax_shap.axvline(x=0, color=COLORS['text_dark'], linewidth=1)
    
    for i, (feat, cont) in enumerate(zip(features, contributions)):
        sign = '+' if cont > 0 else ''
        ax_shap.text(cont + (0.5 if cont >= 0 else -0.5), i, f'{sign}{cont:.1f}',
                    va='center', ha='left' if cont >= 0 else 'right',
                    fontsize=10, fontweight='bold')
    
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
    ax_shap.set_xlabel('Impact on PTP Score', fontsize=11)
    ax_shap.set_title('Key Feature Contributions (SHAP Analysis)', fontsize=12, fontweight='bold')
    ax_shap.spines['top'].set_visible(False)
    ax_shap.spines['right'].set_visible(False)
    
    # 5. Raw Feature Values (bottom left and middle)
    ax_raw = fig.add_subplot(gs[2, :2])
    ax_raw.set_facecolor(COLORS['bg_light'])
    
    # Show top feature values as a table
    top_features = list(feature_values.items())[:12]
    
    # Create table data
    cell_text = [[f.replace('_', ' ').title(), f'{v:.2f}' if isinstance(v, float) else str(v)]
                 for f, v in top_features]
    
    table = ax_raw.table(
        cellText=cell_text,
        colLabels=['Feature', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])
        else:
            cell.set_facecolor(COLORS['bg_light'])
    
    ax_raw.axis('off')
    ax_raw.set_title('Athletic & Production Profile', fontsize=12, fontweight='bold')
    
    # 6. Verdict (bottom right)
    ax_verdict = fig.add_subplot(gs[2, 2])
    ax_verdict.set_facecolor(COLORS['primary'])
    
    # Determine verdict
    if ptp_score >= 70:
        verdict = "HIGH UPSIDE"
        verdict_color = COLORS['success']
        description = "Model projects significant NFL contributor"
    elif ptp_score >= 50:
        verdict = "SOLID PROSPECT"
        verdict_color = COLORS['accent']
        description = "Good chance at meaningful role"
    else:
        verdict = "DEVELOPMENTAL"
        verdict_color = COLORS['warning']
        description = "May need time to develop"
    
    # Check if sleeper
    if draft_round and draft_round >= 5 and ptp_score >= 65:
        verdict = "SLEEPER ALERT"
        verdict_color = COLORS['accent']
        description = f"High value in Round {draft_round}!"
    
    ax_verdict.text(0.5, 0.7, verdict, ha='center', va='center',
                   fontsize=18, fontweight='bold', color=verdict_color,
                   transform=ax_verdict.transAxes)
    ax_verdict.text(0.5, 0.4, description, ha='center', va='center',
                   fontsize=11, color='white', wrap=True,
                   transform=ax_verdict.transAxes)
    
    ax_verdict.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg_light'])
    
    return fig


def create_all_position_leaderboards(
    leaderboard: pl.DataFrame,
    output_dir: Path
) -> Dict[str, plt.Figure]:
    """
    Create leaderboards for all position groups.
    
    Args:
        leaderboard: Full leaderboard DataFrame
        output_dir: Directory to save charts
        
    Returns:
        Dictionary of position group -> figure
    """
    position_groups = {
        'skill': (['WR', 'RB', 'TE', 'FB', 'QB'], 'Skill Positions (WR, RB, TE, QB)'),
        'db': (['DC', 'DS'], 'Defensive Backs (CB, S)'),
        'dl': (['DE', 'DT'], 'Defensive Line (DE, DT)'),
        'lb': (['IB', 'OB'], 'Linebackers (ILB, OLB)'),
        'ol': (['OT', 'OG', 'OC'], 'Offensive Line (OT, OG, C)'),
    }
    
    figures = {}
    
    for group_name, (positions, title) in position_groups.items():
        save_path = output_dir / f'leaderboard_{group_name}.png'
        fig = create_position_leaderboard(
            leaderboard,
            group_name,
            positions,
            f'PTP Score Leaderboard: {title}',
            save_path=save_path,
            top_n=10
        )
        figures[group_name] = fig
        print(f"  Saved: {save_path}")
        plt.close(fig)
    
    return figures


