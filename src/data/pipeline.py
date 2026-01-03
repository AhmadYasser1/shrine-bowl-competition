"""
Data Pipeline for Shrine Bowl Analytics Competition

This module handles loading, linking, and preprocessing all competition datasets:
- Practice tracking data (parquet files)
- Game tracking data (parquet files)
- Player combine/draft data (parquet)
- College stats (CSV)
- NFL rookie stats (CSV)
- Session timestamps (CSV)
"""

import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Union
import os


class DataPipeline:
    """
    Unified data pipeline for loading and linking Shrine Bowl competition data.
    
    All player IDs are standardized to strings for consistent joining.
    """
    
    # Drill types relevant for 1-on-1 analysis
    ONE_ON_ONE_DRILLS = [
        '1 on 1', '1v1', 'Best of 1 on 1', 
        'Big 1 on 1 - Skill 7 on 7', 'Bigs 1 on 1 - Skill 7 on 7',
        'Bigs 9 on 7 - Skill 1 on 1', 'Bigs INDY / Skill 1 on 1',
        'Indy/1v1', '9v7/1v1'
    ]
    
    # Competitive team drills (for broader analysis if 1-on-1 is insufficient)
    TEAM_DRILLS = [
        'Team 1', 'Team 2', 'Team 3', 'Team',
        '7v7', 'Move The Ball', '2 Minute Drill', '2 Min'
    ]
    
    def __init__(self, data_dir: str = "Shrine Bowl Data"):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Path to the Shrine Bowl Data directory
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
        
        # Cache for loaded data
        self._players: Optional[pl.DataFrame] = None
        self._college_stats: Optional[pl.DataFrame] = None
        self._rookie_stats: Optional[pl.DataFrame] = None
        self._session_timestamps: Optional[pl.DataFrame] = None
        
    def _validate_data_dir(self):
        """Validate that required data directories exist."""
        required_paths = [
            self.data_dir / "practice_data",
            self.data_dir / "game_data",
            self.data_dir / "shrine_bowl_players.parquet",
            self.data_dir / "shrine_bowl_players_college_stats.csv",
            self.data_dir / "shrine_bowl_players_nfl_rookie_stats.csv",
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required data path not found: {path}")
    
    # =========================================================================
    # Core Data Loading Methods
    # =========================================================================
    
    def load_players(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load player combine and draft data.
        
        Returns DataFrame with columns:
            - gsis_player_id (str): Unique player identifier
            - Combine metrics: forty_yd_dash, three_cone, standing_broad_jump, etc.
            - Draft info: draft_round, draft_pick, draft_overall_selection
            - Physical: height, weight, arm_length, wingspan, hand_size
        """
        if self._players is None or force_reload:
            df = pl.read_parquet(self.data_dir / "shrine_bowl_players.parquet")
            # Standardize ID to string
            df = df.with_columns(
                pl.col("gsis_player_id").cast(pl.Utf8).alias("player_id")
            )
            self._players = df
        return self._players
    
    def load_college_stats(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load college statistics for all players across seasons.
        
        Returns DataFrame with seasonal stats including:
            - college_gsis_id -> player_id (str)
            - Passing, rushing, receiving stats
            - Defensive stats
            - Return stats
        """
        if self._college_stats is None or force_reload:
            df = pl.read_csv(self.data_dir / "shrine_bowl_players_college_stats.csv")
            # Standardize ID to string
            df = df.with_columns(
                pl.col("college_gsis_id").cast(pl.Utf8).alias("player_id")
            )
            self._college_stats = df
        return self._college_stats
    
    def load_rookie_stats(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load NFL rookie year statistics (SumerSports charted data).
        
        This is the PRIMARY OUTCOME variable for our model.
        
        Returns DataFrame with:
            - college_gsis_id -> player_id (str)
            - total_snaps: Total rookie year snaps (key outcome)
            - Position-specific performance metrics
            - Draft info for validation
        """
        if self._rookie_stats is None or force_reload:
            df = pl.read_csv(self.data_dir / "shrine_bowl_players_nfl_rookie_stats.csv")
            # Standardize ID to string
            df = df.with_columns(
                pl.col("college_gsis_id").cast(pl.Utf8).alias("player_id")
            )
            self._rookie_stats = df
        return self._rookie_stats
    
    def load_session_timestamps(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load session/drill timestamp metadata.
        
        Useful for understanding drill structure and filtering tracking data.
        """
        if self._session_timestamps is None or force_reload:
            df = pl.read_csv(self.data_dir / "session_timestamps.csv")
            self._session_timestamps = df
        return self._session_timestamps
    
    # =========================================================================
    # Tracking Data Loading
    # =========================================================================
    
    def list_practice_files(self) -> List[Path]:
        """List all practice tracking parquet files."""
        practice_dir = self.data_dir / "practice_data"
        return sorted(practice_dir.glob("*.parquet"))
    
    def list_game_files(self) -> List[Path]:
        """List all game tracking parquet files."""
        game_dir = self.data_dir / "game_data"
        return sorted(game_dir.glob("*.parquet"))
    
    def load_practice_tracking(
        self,
        files: Optional[List[str]] = None,
        drill_types: Optional[List[str]] = None,
        entity_type: str = "player",
        lazy: bool = True
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Load practice tracking data with optional filtering.
        
        Args:
            files: Specific files to load (default: all)
            drill_types: Filter to specific drill types (default: all)
            entity_type: 'player', 'ball', or None for both
            lazy: If True, return LazyFrame for memory efficiency
            
        Returns:
            DataFrame/LazyFrame with tracking data:
                - gsis_id -> player_id (str)
                - x, y, z: Position coordinates (yards)
                - s: Speed (yards/second)
                - a: Acceleration (yards/second^2)
                - dir: Movement direction (degrees)
                - sa: Directional acceleration
                - dis: Distance traveled
                - ts: Timestamp
                - drill_type: Name of the drill
        """
        practice_files = self.list_practice_files()
        
        if files:
            practice_files = [f for f in practice_files if f.name in files]
        
        if not practice_files:
            raise ValueError("No practice files found matching criteria")
        
        # Load with lazy evaluation for memory efficiency
        frames = []
        for f in practice_files:
            lf = pl.scan_parquet(f)
            
            # Apply filters
            if entity_type:
                lf = lf.filter(pl.col("entity_type") == entity_type)
            
            if drill_types:
                lf = lf.filter(pl.col("drill_type").is_in(drill_types))
            
            # Standardize player ID
            lf = lf.with_columns(
                pl.col("gsis_id").cast(pl.Utf8).alias("player_id")
            )
            
            # Add source file info
            lf = lf.with_columns(
                pl.lit(f.stem).alias("source_file")
            )
            
            frames.append(lf)
        
        combined = pl.concat(frames)
        
        if lazy:
            return combined
        return combined.collect()
    
    def load_game_tracking(
        self,
        files: Optional[List[str]] = None,
        entity_type: str = "player",
        lazy: bool = True
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Load game tracking data.
        
        Similar to practice tracking but includes play_id for game context.
        """
        game_files = self.list_game_files()
        
        if files:
            game_files = [f for f in game_files if f.name in files]
        
        if not game_files:
            raise ValueError("No game files found matching criteria")
        
        frames = []
        for f in game_files:
            lf = pl.scan_parquet(f)
            
            if entity_type:
                lf = lf.filter(pl.col("entity_type") == entity_type)
            
            lf = lf.with_columns(
                pl.col("gsis_id").cast(pl.Utf8).alias("player_id")
            )
            
            lf = lf.with_columns(
                pl.lit(f.stem).alias("source_file")
            )
            
            frames.append(lf)
        
        combined = pl.concat(frames)
        
        if lazy:
            return combined
        return combined.collect()
    
    def load_one_on_one_drills(self, lazy: bool = True) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Convenience method to load only 1-on-1 drill tracking data.
        
        This is the primary data source for separation/closing metrics.
        """
        return self.load_practice_tracking(
            drill_types=self.ONE_ON_ONE_DRILLS,
            entity_type="player",
            lazy=lazy
        )
    
    # =========================================================================
    # Data Linking
    # =========================================================================
    
    def get_player_outcomes(self) -> pl.DataFrame:
        """
        Create a unified player dataset linking:
        - Player combine/physical data
        - NFL rookie outcomes
        
        This is the base dataset for modeling.
        
        Returns:
            DataFrame with player_id as key, containing:
            - Physical/combine metrics
            - Draft outcomes
            - Rookie year performance
        """
        players = self.load_players()
        rookie_stats = self.load_rookie_stats()
        
        # Select key columns from each source
        player_cols = [
            "player_id", "first_name", "last_name", "football_name",
            "height", "weight", "hand_size", "arm_length", "wingspan",
            "forty_yd_dash", "first_ten_of_forty_yd_dash", 
            "first_twenty_of_forty_yd_dash", "last_twenty_of_forty_yd_dash",
            "three_cone", "twenty_yard_shuffle", 
            "standing_broad_jump", "standing_vertical", "bench_reps_of_225",
            "draft_season", "draft_round", "draft_pick", "draft_overall_selection",
            "team_name", "conference", "recruiting_stars"
        ]
        
        # Get available columns
        available_player_cols = [c for c in player_cols if c in players.columns]
        players_subset = players.select(available_player_cols)
        
        # Key rookie outcome columns
        rookie_cols = [
            "player_id", "rookie_season", "position",
            "total_snaps", "draft_round", "draft_overall_selection",
            # Offensive
            "passing_attempts", "passing_yards", "passing_touchdowns",
            "rushing_attempts", "rushing_yards", "rushing_touchdowns",
            "receiving_targets", "receiving_receptions", "receiving_yards",
            "receiving_yards_per_route_run", "receiving_touchdowns",
            # Defensive
            "pressures", "pressure_rate", "run_defense_tackles",
            "primary_in_coverage_total", "defense_interceptions", "defense_pass_breakups"
        ]
        
        available_rookie_cols = [c for c in rookie_cols if c in rookie_stats.columns]
        rookie_subset = rookie_stats.select(available_rookie_cols)
        
        # Join on player_id
        merged = players_subset.join(
            rookie_subset,
            on="player_id",
            how="inner",
            suffix="_rookie"
        )
        
        return merged
    
    def get_college_career_stats(self) -> pl.DataFrame:
        """
        Aggregate college stats to career-level for each player.
        
        Returns:
            DataFrame with career totals and averages per player.
        """
        college = self.load_college_stats()
        
        # Aggregate numeric columns
        numeric_cols = [
            "passing_attempts", "passing_completions", "passing_yards", "passing_touchdowns",
            "rushing_attempts", "rushing_yards", "rushing_touchdowns",
            "receiving_receptions", "receiving_yards", "receiving_touchdowns",
            "defense_sacks", "defense_total_tackles", "defense_tackles_for_loss",
            "defense_interceptions", "defense_pass_breakups"
        ]
        
        available_numeric = [c for c in numeric_cols if c in college.columns]
        
        # Calculate career totals
        career = college.group_by("player_id").agg([
            pl.col("position").first().alias("position"),
            pl.col("season").max().alias("final_season"),
            pl.col("season").min().alias("first_season"),
            pl.col("season").n_unique().alias("seasons_played"),
        ] + [
            pl.col(c).sum().alias(f"career_{c}") 
            for c in available_numeric if c in college.columns
        ])
        
        return career
    
    def get_tracking_player_ids(self) -> set:
        """
        Get the set of player IDs present in tracking data.
        
        Scans ALL practice files to get complete player coverage.
        Useful for filtering to only players we can analyze.
        """
        practice_files = self.list_practice_files()
        if not practice_files:
            return set()
        
        all_ids = set()
        for f in practice_files:
            df = pl.read_parquet(f)
            ids = df.filter(
                pl.col("gsis_id").is_not_null()
            ).select(
                pl.col("gsis_id").cast(pl.Utf8).alias("player_id")
            ).unique()
            all_ids.update(ids.to_series().to_list())
        
        return all_ids
    
    def get_analyzable_players(self) -> pl.DataFrame:
        """
        Get players who have both tracking data AND rookie outcomes.
        
        These are the players we can use for model training/validation.
        """
        tracking_ids = self.get_tracking_player_ids()
        outcomes = self.get_player_outcomes()
        
        # Filter to players with tracking data
        analyzable = outcomes.filter(
            pl.col("player_id").is_in(list(tracking_ids))
        )
        
        return analyzable


def load_all_data(data_dir: str = "Shrine Bowl Data") -> Dict[str, pl.DataFrame]:
    """
    Convenience function to load all static datasets.
    
    Returns:
        Dictionary with keys: 'players', 'college_stats', 'rookie_stats', 'sessions'
    """
    pipeline = DataPipeline(data_dir)
    
    return {
        'players': pipeline.load_players(),
        'college_stats': pipeline.load_college_stats(),
        'rookie_stats': pipeline.load_rookie_stats(),
        'sessions': pipeline.load_session_timestamps(),
        'player_outcomes': pipeline.get_player_outcomes(),
        'college_career': pipeline.get_college_career_stats(),
    }


if __name__ == "__main__":
    # Quick test of the pipeline
    pipeline = DataPipeline()
    
    print("=== Data Pipeline Test ===")
    print(f"Practice files: {len(pipeline.list_practice_files())}")
    print(f"Game files: {len(pipeline.list_game_files())}")
    
    players = pipeline.load_players()
    print(f"Players loaded: {players.shape}")
    
    rookie = pipeline.load_rookie_stats()
    print(f"Rookie stats loaded: {rookie.shape}")
    
    outcomes = pipeline.get_player_outcomes()
    print(f"Players with outcomes: {outcomes.shape}")
    
    tracking_ids = pipeline.get_tracking_player_ids()
    print(f"Players in tracking data: {len(tracking_ids)}")
    
    analyzable = pipeline.get_analyzable_players()
    print(f"Analyzable players (tracking + outcomes): {analyzable.shape}")

