from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import joblib
import pandas as pd
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Step 4 of Project Spec: Implement comprehensive logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IPL_Backend")

app = FastAPI(title="IPL Prediction Engine Backend")

# Instrument the FastAPI app to expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

MODEL_PATH = "/app/data/models/score_model.pkl"
WIN_MODEL_PATH = "/app/data/models/win_model.pkl"
model_pipeline = None
win_pipeline = None

@app.on_event("startup")
def load_models():
    global model_pipeline
    global win_pipeline
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading score prediction model from {MODEL_PATH}...")
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info("Score model loaded completely.")
    else:
        logger.warning(f"Score model file {MODEL_PATH} not found.")

    if os.path.exists(WIN_MODEL_PATH):
        logger.info(f"Loading win probability model from {WIN_MODEL_PATH}...")
        win_pipeline = joblib.load(WIN_MODEL_PATH)
        logger.info("Win probability model loaded completely.")
    else:
        logger.warning(f"Win probability model file {WIN_MODEL_PATH} not found.")

class MatchState(BaseModel):
    batting_team: str
    bowling_team: str
    current_score: int = Field(ge=0, description="Cumulative score in the inning.")
    wickets_lost: int = Field(ge=0, le=10, description="Must be between 0 and 10 wickets.")
    overs_completed: int = Field(ge=0, le=19, description="Full overs bowled.")
    balls_in_over: int = Field(ge=0, le=5, description="Balls delivered in current over.")

@app.post("/predict_score")
def predict_score(state: MatchState):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction model is not available.")
        
    # Calculate T20 Cricket stats for the model inputs
    total_balls_bowled = (state.overs_completed * 6) + state.balls_in_over
    
    if total_balls_bowled == 0:
        raise HTTPException(status_code=400, detail="Cannot predict before a single ball is bowled.")
        
    balls_left = max(0, 120 - total_balls_bowled)
    current_run_rate = (state.current_score * 6) / total_balls_bowled
    
    input_data = pd.DataFrame([{
        "batting_team": state.batting_team,
        "bowling_team": state.bowling_team,
        "current_score": state.current_score,
        "wickets_lost": state.wickets_lost,
        "balls_left": balls_left,
        "current_run_rate": current_run_rate
    }])
    
    try:
        predicted_score = model_pipeline.predict(input_data)[0]
        # Make sure predicted score isn't smaller than current score
        final_predicted_score = max(state.current_score, int(round(predicted_score)))
        
        logger.info(f"Predicted score: {final_predicted_score} for {state.batting_team} vs {state.bowling_team}")
        return {"predicted_final_score": final_predicted_score}
    except Exception as e:
        logger.error(f"Score prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during score prediction.")

class ChaseState(BaseModel):
    batting_team: str
    bowling_team: str
    current_score: int = Field(ge=0, description="Cumulative score in the inning.")
    wickets_lost: int = Field(ge=0, le=10, description="Must be between 0 and 10 wickets.")
    overs_completed: int = Field(ge=0, le=19, description="Full overs bowled.")
    balls_in_over: int = Field(ge=0, le=5, description="Balls delivered in current over.")
    target_score: int = Field(ge=1, description="The score set by the first batting team.")

@app.post("/predict_win")
def predict_win(state: ChaseState):
    if win_pipeline is None:
        raise HTTPException(status_code=503, detail="Win probability model is not available.")
        
    total_balls_bowled = (state.overs_completed * 6) + state.balls_in_over
    
    if total_balls_bowled == 0:
        raise HTTPException(status_code=400, detail="Cannot predict before a single ball is bowled.")
        
    balls_left = max(0, 120 - total_balls_bowled)
    current_run_rate = (state.current_score * 6) / total_balls_bowled
    runs_required = max(0, state.target_score + 1 - state.current_score)
    required_run_rate = (runs_required * 6) / max(1, balls_left) # Prevent division by zero
    
    input_data = pd.DataFrame([{
        "batting_team": state.batting_team,
        "bowling_team": state.bowling_team,
        "current_score": state.current_score,
        "wickets_lost": state.wickets_lost,
        "balls_left": balls_left,
        "current_run_rate": current_run_rate,
        "target_score": state.target_score,
        "runs_required": runs_required,
        "required_run_rate": required_run_rate
    }])
    
    try:
        # predict_proba returns [[prob_loss, prob_win]] -> Index [0][1] gives the win prob
        win_probability = win_pipeline.predict_proba(input_data)[0][1]
        
        logger.info(f"Predicted Win Prob: {win_probability:.2%} for {state.batting_team} chasing {state.target_score}")
        return {
            "win_probability": round(win_probability * 100, 2),
            "runs_required": runs_required,
            "required_run_rate": round(required_run_rate, 2)
        }
    except Exception as e:
        logger.error(f"Win probability prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during win prediction.")

@app.get("/health")
def health_check():
    """Endpoint for automated orchestration health checks."""
    logger.info("Health check endpoint called.")
    return {"status": "ok"}

@app.get("/ready")
def readiness_check():
    """Endpoint to check if the app is fully ready to accept traffic (e.g., model is loaded)."""
    logger.info("Ready check endpoint called.")
    # In the future, this will check if our ML model is successfully loaded
    return {"status": "ready"}
