import pickle
import streamlit as st
import pandas as pd
import os

# Teams and cities
teams = ['Royal Challengers Bangalore','Punjab Kings','Mumbai Indians','Kolkata Knight Riders',
         'Rajasthan Royals','Chennai Super Kings','Sunrisers Hyderabad','Delhi Capitals',
         'Lucknow Super Giants','Gujarat Titans']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
          'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
          'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Rajkot', 'Kanpur', 'Bengaluru', 'Indore', 'Dubai', 'Sharjah',
          'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']

st.title("IPL Win Predictor")

# Attempt to load the model
MODEL_PATH = 'pipe.pkl'
pipe = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        pipe = pickle.load(f)
else:
    st.warning("Model file not found. Please upload `pipe.pkl`")
    uploaded_file = st.file_uploader("Upload your pickled model here", type=["pkl"])
    if uploaded_file is not None:
        pipe = pickle.load(uploaded_file)
        st.success("Model loaded successfully!")

# Only continue if the model is loaded
if pipe is not None:
    # Team selection
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the batting team', teams)
    with col2:
        bowling_team = st.selectbox('Select the bowling team', teams)

    selected_city = st.selectbox('Select host city', cities)
    target = st.number_input("Target", min_value=0, step=1)

    # Match stats
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score', min_value=0, step=1)
    with col4:
        overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

    # Prediction button
    if st.button('Predict Probability'):
        if overs == 0:
            st.warning("Overs cannot be 0 for probability calculation.")
        else:
            runs_left = target - score
            balls_left = max(0, 120 - (overs * 6))
            wickets_left = 10 - wickets_out
            crr = score / overs
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # Prepare input dataframe
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets_left],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Predict
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            st.subheader("Winning Probability")
            st.text(f"{batting_team} = {round(win*100)}%")
            st.text(f"{bowling_team} = {round(loss*100)}%")
