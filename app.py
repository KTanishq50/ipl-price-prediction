import streamlit as st
import pandas as pd
import joblib

model = joblib.load("player_price_model.pkl")

st.title("IPL Auction Price Predictor")

runs = st.number_input("Runs", value=300)
wickets = st.number_input("Wickets", value=5)
strike_rate = st.number_input("Strike Rate", value=130)
average = st.number_input("Average", value=30)
base_price = st.number_input("Base Price (Lakhs)", value=50)

performance_score = runs + wickets * 20

if st.button("Predict"):
    df = pd.DataFrame([{
        "runs": runs,
        "wickets": wickets,
        "strike_rate": strike_rate,
        "average": average,
        "base_price": base_price,
        "performance_score": performance_score
    }])

    pred = model.predict(df)[0]

    st.success(f"Predicted Price: {round(pred,2)} Lakhs ({round(pred/100,2)} Cr)")
