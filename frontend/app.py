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
            overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1, help="E.g., 14.3")
            runs = st.number_input("Current Runs", min_value=0, max_value=300, step=1)
            
        with col2:
            # Prevent user from selecting the same team for batting and bowling
            bowling_teams = [t for t in [
                "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders", 
                "Mumbai Indians", "Punjab Kings", "Rajasthan Royals", 
                "Royal Challengers Bangalore", "Sunrisers Hyderabad"
            ] if t != batting_team]
            
            bowling_team = st.selectbox("Bowling Team", bowling_teams)
            wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1)

        # Prediction Button
        if st.button("Predict Outcome", type="primary"):
            with st.spinner("Fetching predictions from backend..."):
                try:
                    # In Docker Compose, the backend is reachable via the internal DNS name "backend"
                    # We are calling our /health endpoint just to prove the connection works for now.
                    response = requests.get("http://backend:8000/health")
                    if response.status_code == 200:
                        st.success("Successfully connected to Backend!")
                        
                        # Mock prediction result layout
                        st.subheader("Prediction Results")
                        res_col1, res_col2 = st.columns(2)
                        res_col1.metric("Win Probability", "68%", "+2% from last over")
                        res_col2.metric("Predicted Final Score", "185", "Runs")
                    else:
                        st.error("Backend error.")
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
