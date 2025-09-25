import pickle
import streamlit as st
import pandas as pd

teams= ['Royal Challengers Bangalore',
 'Punjab Kings',
 'Mumbai Indians',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Lucknow Super Giants',
 'Gujarat Titans']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
       'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
       'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Rajkot', 'Kanpur', 'Bengaluru', 'Indore', 'Dubai', 'Sharjah',
       'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']

pipe = pickle.load(open('pipe.pkl' , 'rb'))
st.title("IPL win predictor")

col1 , col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('select the batting team' , teams)
with col2 :
    bowling_team = st.selectbox('select the bowling team' , teams)

selected_city = st.selectbox('select host city' , cities)

target = st.number_input("Target")

col3 , col4 , col5 = st.columns(3)

with col3 :
    score = st.number_input('Score')
with col4:
    overs =  st.number_input('overs completed')
with col5:
    wickets = st.number_input('wickets out')

if st.button('predict probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6 )
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left*6) / balls_left

    input_df = pd.DataFrame({
        'batting_team' : [batting_team] ,
        'bowling_team' : [bowling_team] ,
        'city' : [selected_city] ,
        'runs_left' : [runs_left] ,
        'balls_left' : [balls_left] ,
        'wickets' : [wickets] ,
        'total_runs_x' :[target] ,
        'crr' : [crr] ,
        'rrr' : [rrr]

    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.text("batting_team" + " =  " + str(round(win*100)) + '%')
    st.text("bowling_team" + " =  " + str(round(loss*100))+ '%')

