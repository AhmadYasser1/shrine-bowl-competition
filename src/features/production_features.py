"""
College Production Feature Extraction for Shrine Bowl Analytics

Aggregates college statistics into predictive features:
- Career totals and per-game averages
- Dominator rating (share of team production)
- Breakout age/season
- Production trajectory (improvement trends)

These features capture college-level performance context.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional


class ProductionFeatureExtractor:
    """
    Extract production features from college statistics.
    
    Key metrics:
    - career_totals: Cumulative stats across college career
    - per_game_rates: Production normalized by games/seasons
    - dominator_rating: Proxy for share of team production
    - career_trajectory: Trend in production over time
    """
    
    # Stat columns by position type
    OFFENSIVE_SKILL_STATS = [
        "receiving_receptions",
        "receiving_yards", 
        "receiving_touchdowns",
        "rushing_attempts",
        "rushing_yards",
        "rushing_touchdowns"
    ]
    
    PASSING_STATS = [
        "passing_attempts",
        "passing_completions",
        "passing_yards",
        "passing_touchdowns",
        "passing_interceptions"
    ]
    
    DEFENSIVE_STATS = [
        "defense_total_tackles",
        "defense_solo_tackles",
        "defense_tackles_for_loss",
        "defense_sacks",
        "defense_interceptions",
        "defense_pass_breakups"
    ]
    
    def __init__(self):
        """Initialize the production feature extractor."""
        pass
    
    def extract_features(
        self,
        college_stats: pl.DataFrame,
        player_ids: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Extract production features for players.
        
        Args:
            college_stats: Season-by-season college statistics
            player_ids: Optional filter for specific players
            
        Returns:
            DataFrame with player_id and production features
        """
        if player_ids:
            college_stats = college_stats.filter(
                pl.col("player_id").is_in(player_ids)
            )
        
        # Get unique players
        unique_players = college_stats.select("player_id").unique()
        
        all_features = []
        
        for row in unique_players.iter_rows():
            player_id = row[0]
            player_stats = college_stats.filter(pl.col("player_id") == player_id)
            
            features = {"player_id": player_id}
            
            # Career totals
            features.update(self._extract_career_totals(player_stats))
            
            # Production rates
            features.update(self._extract_production_rates(player_stats))
            
            # Career trajectory
            features.update(self._extract_trajectory(player_stats))
            
            # Dominator rating proxy
            features.update(self._extract_dominator_metrics(player_stats))
            
            all_features.append(features)
        
        return pl.DataFrame(all_features)
    
    def _extract_career_totals(self, player_stats: pl.DataFrame) -> Dict[str, float]:
        """Sum statistics across all college seasons."""
        features = {}
        
        # Count seasons
        features["college_seasons"] = float(len(player_stats))
        
        # Get position (take most recent)
        if "position" in player_stats.columns:
            positions = player_stats.select("position").to_series().to_list()
            features["college_position"] = positions[-1] if positions else "UNK"
        
        # Offensive skill totals
        for col in self.OFFENSIVE_SKILL_STATS:
            if col in player_stats.columns:
                total = player_stats.select(col).sum().item()
                features[f"career_{col}"] = float(total) if total is not None else 0.0
        
        # Passing totals
        for col in self.PASSING_STATS:
            if col in player_stats.columns:
                total = player_stats.select(col).sum().item()
                features[f"career_{col}"] = float(total) if total is not None else 0.0
        
        # Defensive totals
        for col in self.DEFENSIVE_STATS:
            if col in player_stats.columns:
                total = player_stats.select(col).sum().item()
                features[f"career_{col}"] = float(total) if total is not None else 0.0
        
        return features
    
    def _extract_production_rates(self, player_stats: pl.DataFrame) -> Dict[str, float]:
        """Calculate per-season production rates."""
        features = {}
        n_seasons = len(player_stats)
        
        if n_seasons == 0:
            return features
        
        # Receiving yards per season
        if "receiving_yards" in player_stats.columns:
            total = player_stats.select("receiving_yards").sum().item()
            features["rec_yards_per_season"] = float(total or 0) / n_seasons
        
        # Rushing yards per season  
        if "rushing_yards" in player_stats.columns:
            total = player_stats.select("rushing_yards").sum().item()
            features["rush_yards_per_season"] = float(total or 0) / n_seasons
        
        # Total scrimmage yards per season
        rec = player_stats.select("receiving_yards").sum().item() or 0 if "receiving_yards" in player_stats.columns else 0
        rush = player_stats.select("rushing_yards").sum().item() or 0 if "rushing_yards" in player_stats.columns else 0
        features["scrimmage_yards_per_season"] = float(rec + rush) / n_seasons
        
        # TDs per season
        rec_td = player_stats.select("receiving_touchdowns").sum().item() or 0 if "receiving_touchdowns" in player_stats.columns else 0
        rush_td = player_stats.select("rushing_touchdowns").sum().item() or 0 if "rushing_touchdowns" in player_stats.columns else 0
        features["tds_per_season"] = float(rec_td + rush_td) / n_seasons
        
        # Tackles per season (for defenders)
        if "defense_total_tackles" in player_stats.columns:
            total = player_stats.select("defense_total_tackles").sum().item()
            features["tackles_per_season"] = float(total or 0) / n_seasons
        
        # Sacks per season
        if "defense_sacks" in player_stats.columns:
            total = player_stats.select("defense_sacks").sum().item()
            features["sacks_per_season"] = float(total or 0) / n_seasons
        
        return features
    
    def _extract_trajectory(self, player_stats: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate production trajectory (improvement over time).
        
        Positive trajectory = improving each year
        """
        features = {}
        
        if len(player_stats) < 2:
            features["production_trajectory"] = 0.0
            features["peak_season_year"] = 0
            return features
        
        # Sort by season
        player_stats = player_stats.sort("season")
        seasons = player_stats.select("season").to_series().to_numpy()
        
        # Calculate total production per season
        production = np.zeros(len(seasons))
        
        for i in range(len(seasons)):
            season_stats = player_stats.row(i, named=True)
            
            # Sum key production metrics
            prod = 0
            for col in ["receiving_yards", "rushing_yards", "defense_total_tackles"]:
                if col in season_stats and season_stats[col] is not None:
                    prod += season_stats[col]
            
            production[i] = prod
        
        # Calculate trajectory (slope of production over time)
        if len(production) >= 2 and np.std(production) > 0:
            # Simple linear regression slope
            x = np.arange(len(production))
            slope = np.polyfit(x, production, 1)[0]
            features["production_trajectory"] = float(slope)
        else:
            features["production_trajectory"] = 0.0
        
        # Find peak season
        if len(production) > 0:
            peak_idx = np.argmax(production)
            # Express as years from final season (0 = final year, -1 = second to last, etc.)
            features["peak_season_year"] = float(peak_idx - len(production) + 1)
        else:
            features["peak_season_year"] = 0.0
        
        return features
    
    def _extract_dominator_metrics(self, player_stats: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate dominator rating proxies.
        
        True dominator rating requires team totals (which we don't have),
        so we use proxies based on individual production levels.
        """
        features = {}
        
        if len(player_stats) == 0:
            features["dominator_proxy"] = 0.0
            return features
        
        # For skill players: combine yards and TDs
        rec_yards = player_stats.select("receiving_yards").sum().item() or 0 if "receiving_yards" in player_stats.columns else 0
        rec_td = player_stats.select("receiving_touchdowns").sum().item() or 0 if "receiving_touchdowns" in player_stats.columns else 0
        rush_yards = player_stats.select("rushing_yards").sum().item() or 0 if "rushing_yards" in player_stats.columns else 0
        rush_td = player_stats.select("rushing_touchdowns").sum().item() or 0 if "rushing_touchdowns" in player_stats.columns else 0
        
        # Dominator proxy for skill players
        # Weights: Yards (1 point per 100), TDs (10 points each)
        skill_dominator = (rec_yards + rush_yards) / 100 + (rec_td + rush_td) * 10
        
        # For defenders: combine tackles and impact plays
        tackles = player_stats.select("defense_total_tackles").sum().item() or 0 if "defense_total_tackles" in player_stats.columns else 0
        sacks = player_stats.select("defense_sacks").sum().item() or 0 if "defense_sacks" in player_stats.columns else 0
        ints = player_stats.select("defense_interceptions").sum().item() or 0 if "defense_interceptions" in player_stats.columns else 0
        tfl = player_stats.select("defense_tackles_for_loss").sum().item() or 0 if "defense_tackles_for_loss" in player_stats.columns else 0
        
        # Defender dominator proxy
        def_dominator = tackles / 10 + sacks * 5 + ints * 10 + tfl * 3
        
        # Take max of skill and def (player should only have one)
        features["dominator_proxy"] = float(max(skill_dominator, def_dominator))
        
        # Normalized per season
        n_seasons = len(player_stats)
        features["dominator_per_season"] = features["dominator_proxy"] / n_seasons if n_seasons > 0 else 0.0
        
        return features


def extract_production_features_for_modeling(
    pipeline,
    analyzable_player_ids: List[str]
) -> pl.DataFrame:
    """
    Convenience function to extract production features for modeling.
    
    Args:
        pipeline: DataPipeline instance
        analyzable_player_ids: List of player IDs to extract features for
        
    Returns:
        DataFrame with player_id and production features
    """
    college_stats = pipeline.load_college_stats()
    
    # Extract features
    extractor = ProductionFeatureExtractor()
    features = extractor.extract_features(college_stats, analyzable_player_ids)
    
    return features


if __name__ == "__main__":
    # Test the feature extractor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.pipeline import DataPipeline
    
    pipeline = DataPipeline()
    
    # Get analyzable players
    analyzable = pipeline.get_analyzable_players()
    player_ids = analyzable.select("player_id").to_series().to_list()
    
    print(f"Testing feature extraction for {len(player_ids)} players...")
    
    features = extract_production_features_for_modeling(pipeline, player_ids)
    
    print("\nExtracted production features:")
    print(features.head(10))
    print(f"\nShape: {features.shape}")
    print(f"\nColumns: {features.columns}")


