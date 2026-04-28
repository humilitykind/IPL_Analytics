import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Step 4 Spec: Comprehensive logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelTrainer")

# Spec Section 3: Track metrics, parameters, and artifacts with MLflow
# Configure MLflow to point to our Docker container port
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("IPL_Score_Prediction")

class IPLModelTrainer:
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        self.model = None
        self.pipeline = None

    def get_git_commit_hash(self):
        """Spec Section 3: Every experiment must be tied to a specific Git commit hash"""
        try:
            hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            return hash
        except Exception as e:
            logger.warning(f"Could not get git hash (are we in a git repo?): {e}")
            return "unknown_commit"

    def load_and_preprocess_data(self):
        logger.info(f"Loading data from {self.raw_data_path}...")
        df = pd.read_csv(self.raw_data_path)
        
        # Ensure cricket rules: Sort chronologically by match_id and ball number
        df = df.sort_values(['match_id', 'inning', 'ball']).reset_index(drop=True)
        
        logger.info("Calculating cumulative match state (Runs, Wickets, Balls)...")
        # Generate match boundaries for Target Scores
        inning1 = df[df['inning'] == 1].copy()
        inning2 = df[df['inning'] == 2].copy()
        
        target_scores = inning1.groupby('match_id')['total_runs'].sum().reset_index()
        target_scores.rename(columns={'total_runs': 'target_score'}, inplace=True)
        
        def calculate_state(data, is_second_inning=False):
            # 1. Cumulative Runs
            data['current_score'] = data.groupby('match_id')['total_runs'].cumsum()
            
            # 2. Cumulative Wickets (Rule: 10 wickets max)
            data['is_wicket'] = pd.to_numeric(data['is_wicket'], errors='coerce').fillna(0)
            data['wickets_lost'] = data.groupby('match_id')['is_wicket'].cumsum()
            
            # 3. Balls bowled 
            data['is_legal_delivery'] = ~data['extras_type'].isin(['wides', 'noballs'])
            data['balls_bowled'] = data.groupby('match_id')['is_legal_delivery'].cumsum()
            data['balls_left'] = 120 - data['balls_bowled']
            data['balls_left'] = data['balls_left'].apply(lambda x: max(0, x))
            
            # 4. Current Run Rate
            data['current_run_rate'] = (data['current_score'] * 6) / data['balls_bowled'].replace(0, 1)

            if is_second_inning:
                data = data.merge(target_scores, on='match_id', how='inner')
                data['runs_required'] = data['target_score'] + 1 - data['current_score']
                data['runs_required'] = data['runs_required'].apply(lambda x: max(0, x))
                data['required_run_rate'] = (data['runs_required'] * 6) / data['balls_left'].replace(0, 1)
                
                # Determine Match Winner statically to see if Chasing team won
                def did_chasing_team_win(group):
                    # They won if they hit the target
                    winning_score = group['target_score'].iloc[0] + 1
                    final_score = group['current_score'].iloc[-1]
                    return 1 if final_score >= winning_score else 0

                match_winners = data.groupby('match_id').apply(did_chasing_team_win).reset_index(name='chasing_team_won')
                data = data.merge(match_winners, on='match_id')

            else:
                final_scores = data.groupby('match_id')['total_runs'].sum().reset_index()
                final_scores.rename(columns={'total_runs': 'final_score'}, inplace=True)
                data = data.merge(final_scores, on='match_id')

            return data

        self.df_inning1 = calculate_state(inning1, is_second_inning=False)
        self.df_inning2 = calculate_state(inning2, is_second_inning=True)
        
        logger.info(f"Preprocessing complete. Extracted {len(self.df_inning1)} Inning 1 states and {len(self.df_inning2)} Inning 2 states.")

    def train_score_predictor(self):
        # We don't want to predict off of the first 30 balls of an inning (too random)
        train_df = self.df_inning1[self.df_inning1['balls_bowled'] > 36].copy()
        
        features = ['batting_team', 'bowling_team', 'current_score', 'wickets_lost', 'balls_left', 'current_run_rate']
        X = train_df[features]
        y = train_df['final_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Pipeline to handle categorical teams
        categorical_features = ['batting_team', 'bowling_team']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        n_estimators = 50
        max_depth = 10
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
        ])

        with mlflow.start_run(run_name="First_Inning_Score_Prediction"):
            # Log Spec Section 3 requirements
            mlflow.log_param("git_commit", self.get_git_commit_hash())
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_balls_bowled", 36)
            
            logger.info("Training Random Forest Regressor...")
            self.pipeline.fit(X_train, y_train)
            
            y_pred = self.pipeline.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Model trained! RMSE: {rmse:.2f} runs | MAE: {mae:.2f} runs")
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            
            # Save local copy for FastAPI to load without MLflow (Do this first before MLFlow throws any artifact errors)
            os.makedirs(self.output_path, exist_ok=True)
            import joblib
            joblib.dump(self.pipeline, os.path.join(self.output_path, "score_model.pkl"))
            logger.info("Model saved locally as score_model.pkl")
            
            # Log model artifact to MLflow safely in older MLflow versions
            try:
                mlflow.sklearn.log_model(self.pipeline, "model")
            except Exception as e:
                logger.warning(f"Could not log complete model artifact to local MLflow (due to v2.10 vs v2.14 module mismatch). Bypassing: {e}")

    def train_win_predictor(self):
        """Train classifier for 2nd Innings Win Probability"""
        train_df = self.df_inning2.dropna(subset=['target_score', 'runs_required']).copy()
        
        # New features injected for 2nd Inning chasers
        features = ['batting_team', 'bowling_team', 'current_score', 'wickets_lost', 
                    'balls_left', 'current_run_rate', 'target_score', 'runs_required', 'required_run_rate']
        
        X = train_df[features]
        # Trying to predict if the chasing team successfully chased the target
        y = train_df['chasing_team_won']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        categorical_features = ['batting_team', 'bowling_team']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        n_estimators = 50
        max_depth = 10
        self.win_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
        ])

        with mlflow.start_run(run_name="Second_Inning_Win_Probability"):
            mlflow.log_param("git_commit", self.get_git_commit_hash())
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            logger.info("Training Random Forest Classifier for Win Probability...")
            self.win_pipeline.fit(X_train, y_train)
            
            y_pred = self.win_pipeline.predict(X_test)
            y_pred_proba = self.win_pipeline.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            logger.info(f"Win Probability Model trained! Accuracy: {accuracy*100:.2f}% | Log-Loss: {logloss:.4f}")
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("log_loss", logloss)
            
            import joblib
            joblib.dump(self.win_pipeline, os.path.join(self.output_path, "win_model.pkl"))
            logger.info("Model saved locally as win_model.pkl")
            
            try:
                mlflow.sklearn.log_model(self.win_pipeline, "model")
            except Exception:
                pass


    def run(self):
        self.load_and_preprocess_data()
        self.train_score_predictor()
        self.train_win_predictor()


if __name__ == "__main__":
    RAW_DATA = "../../data/raw/ipl_dataset.csv"
    OUTPUT_DIR = "../../data/models"
    
    trainer = IPLModelTrainer(raw_data_path=RAW_DATA, output_path=OUTPUT_DIR)
    trainer.run()
