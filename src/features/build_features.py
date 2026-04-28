import pandas as pd
import logging
import os

# Step 4 Spec: Comprehensive logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureEngineer")

class IPLFeatureEngineer:
    """
    Object-Oriented paradigm (Spec Section 1) to encapsulate feature engineering logic separately from model logic (Spec Section 2).
    """
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        self.batter_stats = None
        self.bowler_stats = None

    def load_data(self):
        logger.info(f"Loading data from {self.raw_data_path}...")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        
    def calculate_batter_baselines(self):
        """
        Circumvent player name cardinality by calculating statistical baselines (Spec Section 2).
        Calculates: Career Strike Rate
        """
        logger.info("Calculating batter baselines (Strike Rates)...")
        # Total runs scored by batter
        runs_scored = self.df.groupby('batter')['batsman_runs'].sum()
        # Total balls faced (excluding wide balls)
        balls_faced = self.df[~self.df['extras_type'].isin(['wides'])].groupby('batter')['ball'].count()
        
        self.batter_stats = pd.DataFrame({
            'total_runs': runs_scored,
            'balls_faced': balls_faced
        }).fillna(0)
        
        # Avoid division by zero
        self.batter_stats['career_strike_rate'] = (
            self.batter_stats['total_runs'] / self.batter_stats['balls_faced'].replace(0, 1)
        ) * 100
        
        logger.info(f"Batter stats generated for {len(self.batter_stats)} players.")

    def calculate_bowler_baselines(self):
        """
        Calculates: Bowling Economy (Runs conceded per over)
        """
        logger.info("Calculating bowler baselines (Economy Rates)...")
        
        # Exclude legbyes and byes (they don't count against the bowler's runs)
        valid_bowling_deliveries = self.df[~self.df['extras_type'].isin(['legbyes', 'byes'])]
        
        runs_conceded = valid_bowling_deliveries.groupby('bowler')['total_runs'].sum()
        
        # Legal deliveries (excluding wides and noballs)
        legal_deliveries = self.df[~self.df['extras_type'].isin(['wides', 'noballs'])].groupby('bowler')['ball'].count()
        overs_bowled = legal_deliveries / 6.0
        
        self.bowler_stats = pd.DataFrame({
            'runs_conceded': runs_conceded,
            'overs_bowled': overs_bowled
        }).fillna(0)
        
        # Avoid division by zero
        self.bowler_stats['bowling_economy'] = (
            self.bowler_stats['runs_conceded'] / self.bowler_stats['overs_bowled'].replace(0, 1)
        )
        
        logger.info(f"Bowler stats generated for {len(self.bowler_stats)} players.")

    def save_baselines(self):
        """
        Calculate and store baseline statistics (mean, variance) for future drift detection (Spec Section 2).
        """
        logger.info(f"Saving feature engineering outputs to {self.output_path}...")
        os.makedirs(self.output_path, exist_ok=True)
        
        if self.batter_stats is not None:
            self.batter_stats.to_csv(os.path.join(self.output_path, "batter_baselines.csv"))
            
        if self.bowler_stats is not None:
            self.bowler_stats.to_csv(os.path.join(self.output_path, "bowler_baselines.csv"))
            
        logger.info("Feature engineering and baseline savings completed successfully.")

    def run_pipeline(self):
        self.load_data()
        self.calculate_batter_baselines()
        self.calculate_bowler_baselines()
        self.save_baselines()

if __name__ == "__main__":
    # Define paths
    RAW_DATA = "../../data/raw/ipl_dataset.csv"
    PROCESSED_DATA_DIR = "../../data/processed"
    
    engineer = IPLFeatureEngineer(raw_data_path=RAW_DATA, output_path=PROCESSED_DATA_DIR)
    engineer.run_pipeline()
