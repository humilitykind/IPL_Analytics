import streamlit as st
import requests

# Configure page settings
st.set_page_config(page_title="IPL Prediction Engine", layout="wide")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction UI", "ML Pipeline Console"])

    if page == "Prediction UI":
        st.title("🏏 IPL Win Probability & Score Prediction")
        st.write("Enter the current match state to get real-time predictions.")

        # User Inputs (Foolproof UI)
        col1, col2 = st.columns(2)
        
        with col1:
            batting_team = st.selectbox("Batting Team", [
                "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders", 
                "Mumbai Indians", "Punjab Kings", "Rajasthan Royals", 
                "Royal Challengers Bangalore", "Sunrisers Hyderabad"
            ])
            runs = st.number_input("Current Score", min_value=0, max_value=300, step=1)
            
        with col2:
            # Prevent user from selecting the same team for batting and bowling
            bowling_teams = [t for t in [
                "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders", 
                "Mumbai Indians", "Punjab Kings", "Rajasthan Royals", 
                "Royal Challengers Bangalore", "Sunrisers Hyderabad"
            ] if t != batting_team]
            
            bowling_team = st.selectbox("Bowling Team", bowling_teams)
            wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1)

        # Inning toggle and target score
        inning_type = st.radio("Which inning are we currently in?", ["1st Inning (Setting Score)", "2nd Inning (Chasing Target)"])
        
        target_score = 0
        if inning_type == "2nd Inning (Chasing Target)":
            st.info(f"Team '{batting_team}' is chasing.")
            target_score = st.number_input("What is the Target Score?", min_value=1, max_value=350, step=1)

        # Enforcing T20 Rules: Separating overs from balls in over
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            overs_completed = st.number_input("Completed Overs", min_value=0, max_value=19, step=1)
        with o_col2:
            balls_in_over = st.number_input("Balls bowled in current over", min_value=0, max_value=5, step=1)

        # Wait until at least 1 legitimate ball is bowled to predict
        if overs_completed == 0 and balls_in_over == 0:
            st.warning("Please enter at least 1 legal ball bowled.")
        
        elif inning_type == "2nd Inning (Chasing Target)" and runs > target_score:
            st.error("Current score cannot be greater than the target score (the match implies they already won!).")

        # Prediction Button
        elif st.button("Predict Outcome", type="primary"):
            with st.spinner("Calculating ML prediction..."):
                try:
                    if inning_type == "1st Inning (Setting Score)":
                        payload = {
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "current_score": runs,
                            "wickets_lost": wickets,
                            "overs_completed": overs_completed,
                            "balls_in_over": balls_in_over
                        }
                        response = requests.post("http://backend:8000/predict_score", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            predicted_score = data["predicted_final_score"]
                            st.success("Successfully calculated Final Score Prediction!")
                            st.subheader("1st Inning Prediction")
                            res_col1, res_col2 = st.columns(2)
                            res_col1.metric("Predicted Final Score", str(predicted_score) + " Runs")
                            res_col2.metric("Current Run Rate", f"{(runs * 6) / ((overs_completed * 6) + balls_in_over):.2f}")
                        else:
                            st.error(f"Backend error: {response.text}")

                    elif inning_type == "2nd Inning (Chasing Target)":
                        payload = {
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "current_score": runs,
                            "wickets_lost": wickets,
                            "overs_completed": overs_completed,
                            "balls_in_over": balls_in_over,
                            "target_score": target_score
                        }
                        response = requests.post("http://backend:8000/predict_win", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            win_prob = data["win_probability"]
                            runs_req = data["runs_required"]
                            req_rr = data["required_run_rate"]
                            
                            st.success(f"Successfully calculated Win Probability for {batting_team}!")
                            st.subheader("2nd Inning Prediction")
                            res_col1, res_col2, res_col3 = st.columns(3)
                            
                            # Displaying probability explicitly
                            if win_prob > 50:
                                prob_delta = f"Favorite to Win"
                            else:
                                prob_delta = f"Unlikely to Win"
                                
                            res_col1.metric(f"Win Probability ({batting_team})", f"{win_prob}%", prob_delta)
                            res_col2.metric("Runs Required", str(runs_req))
                            res_col3.metric("Required Run Rate", str(req_rr))
                        else:
                            st.error(f"Backend error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Is it running?")

    elif page == "ML Pipeline Console":
        st.title("⚙️ ML Pipeline Management Console")
        st.write("Track successful runs, failures, data drift, and data ingestion from Apache Airflow/Spark.")
        
        # Placeholder for Pipeline Visibility (Section 5 of Specs)
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Serving Status", "Healthy")
        col2.metric("Last Data Ingestion", "2 Hrs Ago")
        col3.metric("Data Drift Detected", "No", delta_color="off")
        
        st.info("Continuous Training Pipeline visualization will be embedded here.")

if __name__ == "__main__":
    main()
