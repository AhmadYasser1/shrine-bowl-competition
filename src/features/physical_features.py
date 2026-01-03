"""
Physical/Athletic Feature Extraction for Shrine Bowl Analytics

Processes combine and physical testing data into normalized features:
- Athleticism indices (z-scored by position group)
- Size-speed composites
- Movement efficiency scores

These features provide context for tracking-derived metrics.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional


class PhysicalFeatureExtractor:
    """
    Extract and normalize physical/athletic features from combine data.
    
    Key metrics:
    - athleticism_index: Composite z-score of athletic tests
    - size_speed_score: Speed relative to body mass
    - explosion_index: Vertical + broad jump composite
    - agility_index: 3-cone + shuttle composite
    """
    
    # Athletic testing columns
    SPEED_COLS = [
        "forty_yd_dash",
        "first_ten_of_forty_yd_dash",
        "first_twenty_of_forty_yd_dash",
        "last_twenty_of_forty_yd_dash"
    ]
    
    AGILITY_COLS = [
        "three_cone",
        "twenty_yard_shuffle"
    ]
    
    EXPLOSION_COLS = [
        "standing_broad_jump",
        "standing_vertical"
    ]
    
    STRENGTH_COLS = [
        "bench_reps_of_225"
    ]
    
    SIZE_COLS = [
        "height",
        "weight",
        "arm_length",
        "wingspan",
        "hand_size"
    ]
    
    # Position groups for normalization
    POSITION_GROUPS = {
        "skill": ["WR", "RB", "TE", "FB", "QB"],
        "db": ["DC", "DS"],
        "lb": ["IB", "OB"],
        "dl": ["DE", "DT"],
        "ol": ["OT", "OG", "OC"]
    }
    
    def __init__(self):
        """Initialize the physical feature extractor."""
        self._position_stats = None  # Cached position group statistics
    
    def extract_features(
        self,
        players_df: pl.DataFrame,
        normalize_by_position: bool = True
    ) -> pl.DataFrame:
        """
        Extract physical features for all players.
        
        Args:
            players_df: Player data with physical measurements
            normalize_by_position: If True, z-score within position groups
            
        Returns:
            DataFrame with player_id and physical features
        """
        # Start with player ID
        result = players_df.select("player_id")
        
        # Raw physical metrics
        result = self._add_raw_features(players_df, result)
        
        # Composite indices
        result = self._add_composite_indices(players_df, result)
        
        # Normalize by position if requested
        if normalize_by_position and "position" in players_df.columns:
            result = self._normalize_by_position(players_df, result)
        
        return result
    
    def _add_raw_features(
        self,
        source: pl.DataFrame,
        result: pl.DataFrame
    ) -> pl.DataFrame:
        """Add raw physical measurements to result."""
        # Add size metrics (cast to float in case they're strings)
        for col in self.SIZE_COLS:
            if col in source.columns:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series()
                result = result.with_columns(vals.alias(col))
        
        # Add 40-yard dash (invert so higher is better for correlation)
        if "forty_yd_dash" in source.columns:
            forty = source.select(
                pl.col("forty_yd_dash").cast(pl.Float64, strict=False)
            ).to_series()
            # Speed score: lower time = higher score
            result = result.with_columns(
                forty.alias("forty_yd_dash"),
                (10.0 - forty).alias("speed_score")  # Invert so higher is better
            )
        
        # Add agility times (cast to float)
        for col in self.AGILITY_COLS:
            if col in source.columns:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series()
                result = result.with_columns(vals.alias(col))
        
        # Add explosion metrics (cast to float)
        for col in self.EXPLOSION_COLS:
            if col in source.columns:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series()
                result = result.with_columns(vals.alias(col))
        
        # Add strength (cast to float)
        for col in self.STRENGTH_COLS:
            if col in source.columns:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series()
                result = result.with_columns(vals.alias(col))
        
        return result
    
    def _add_composite_indices(
        self,
        source: pl.DataFrame,
        result: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate composite athletic indices."""
        
        # Size-Speed Score: Speed relative to weight
        # Formula: (10 - forty_time) * (weight / 200)
        if "forty_yd_dash" in source.columns and "weight" in source.columns:
            forty = source.select(
                pl.col("forty_yd_dash").cast(pl.Float64, strict=False)
            ).to_series().to_numpy()
            weight = source.select(
                pl.col("weight").cast(pl.Float64, strict=False)
            ).to_series().to_numpy()
            
            # Higher weight at same speed = more impressive
            size_speed = (10.0 - forty) * (weight / 200.0)
            size_speed = np.nan_to_num(size_speed, nan=np.nan)
            
            result = result.with_columns(
                pl.Series("size_speed_score", size_speed)
            )
        
        # Explosion Index: Normalized sum of jumps
        jump_cols = [c for c in self.EXPLOSION_COLS if c in source.columns]
        if len(jump_cols) > 0:
            explosion_scores = []
            for col in jump_cols:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series().to_numpy()
                # Z-score
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                if std > 0:
                    z = (vals - mean) / std
                else:
                    z = np.zeros_like(vals)
                explosion_scores.append(z)
            
            # Average z-scores
            explosion_index = np.nanmean(explosion_scores, axis=0)
            result = result.with_columns(
                pl.Series("explosion_index", explosion_index)
            )
        
        # Agility Index: Normalized sum of agility times (inverted)
        agility_cols = [c for c in self.AGILITY_COLS if c in source.columns]
        if len(agility_cols) > 0:
            agility_scores = []
            for col in agility_cols:
                vals = source.select(
                    pl.col(col).cast(pl.Float64, strict=False)
                ).to_series().to_numpy()
                # Z-score (inverted - lower time is better)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                if std > 0:
                    z = -(vals - mean) / std  # Negative because lower is better
                else:
                    z = np.zeros_like(vals)
                agility_scores.append(z)
            
            agility_index = np.nanmean(agility_scores, axis=0)
            result = result.with_columns(
                pl.Series("agility_index", agility_index)
            )
        
        # Overall Athleticism Index: Combination of all athletic tests
        # Weight: Speed (0.4), Explosion (0.3), Agility (0.3)
        athleticism_components = []
        weights = []
        
        if "speed_score" in result.columns:
            speed = result.select("speed_score").to_series().to_numpy()
            # Z-score the speed score
            mean = np.nanmean(speed)
            std = np.nanstd(speed)
            if std > 0:
                speed_z = (speed - mean) / std
                athleticism_components.append(speed_z)
                weights.append(0.4)
        
        if "explosion_index" in result.columns:
            explosion = result.select("explosion_index").to_series().to_numpy()
            athleticism_components.append(explosion)
            weights.append(0.3)
        
        if "agility_index" in result.columns:
            agility = result.select("agility_index").to_series().to_numpy()
            athleticism_components.append(agility)
            weights.append(0.3)
        
        if len(athleticism_components) > 0:
            # Weighted average
            weights = np.array(weights) / np.sum(weights)
            athleticism_index = np.zeros(len(result))
            for comp, w in zip(athleticism_components, weights):
                athleticism_index += w * np.nan_to_num(comp, nan=0.0)
            
            result = result.with_columns(
                pl.Series("athleticism_index", athleticism_index)
            )
        
        return result
    
    def _normalize_by_position(
        self,
        source: pl.DataFrame,
        result: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Add position-normalized versions of key metrics.
        
        This allows comparing a WR's speed to other WRs rather than to OL.
        """
        if "position" not in source.columns:
            return result
        
        position = source.select("position").to_series()
        
        # Determine position group for each player
        def get_position_group(pos):
            for group, positions in self.POSITION_GROUPS.items():
                if pos in positions:
                    return group
            return "other"
        
        pos_groups = [get_position_group(p) for p in position.to_list()]
        result = result.with_columns(
            pl.Series("position_group", pos_groups)
        )
        
        # Normalize athleticism index by position group
        if "athleticism_index" in result.columns:
            ath = result.select("athleticism_index").to_series().to_numpy()
            ath_normalized = np.zeros_like(ath)
            
            for group in self.POSITION_GROUPS.keys():
                mask = np.array(pos_groups) == group
                if np.sum(mask) > 1:
                    group_vals = ath[mask]
                    mean = np.nanmean(group_vals)
                    std = np.nanstd(group_vals)
                    if std > 0:
                        ath_normalized[mask] = (group_vals - mean) / std
                    else:
                        ath_normalized[mask] = 0.0
            
            result = result.with_columns(
                pl.Series("athleticism_index_pos_adj", ath_normalized)
            )
        
        return result


def extract_physical_features_for_modeling(
    pipeline,
    analyzable_player_ids: List[str]
) -> pl.DataFrame:
    """
    Convenience function to extract physical features for modeling.
    
    Args:
        pipeline: DataPipeline instance
        analyzable_player_ids: List of player IDs to extract features for
        
    Returns:
        DataFrame with player_id and physical features
    """
    players = pipeline.load_players()
    
    # Filter to analyzable players
    players = players.filter(
        pl.col("player_id").is_in(analyzable_player_ids)
    )
    
    # Get position from rookie stats for normalization
    rookie = pipeline.load_rookie_stats()
    rookie = rookie.filter(
        pl.col("player_id").is_in(analyzable_player_ids)
    ).select(["player_id", "position"])
    
    # Join position to players
    players = players.join(rookie, on="player_id", how="left")
    
    # Extract features
    extractor = PhysicalFeatureExtractor()
    features = extractor.extract_features(players, normalize_by_position=True)
    
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
    
    features = extract_physical_features_for_modeling(pipeline, player_ids)
    
    print("\nExtracted physical features:")
    print(features.head(10))
    print(f"\nShape: {features.shape}")

