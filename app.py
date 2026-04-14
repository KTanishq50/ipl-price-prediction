import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("player_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("IPL Auction Price Predictor")

# float inputs
runs = st.number_input("Runs", value=300.0, step=1.0)
wickets = st.number_input("Wickets", value=5.0, step=1.0)
strike_rate = st.number_input("Strike Rate", value=130.0, step=0.1)
average = st.number_input("Average", value=30.0, step=0.1)
base_price = st.number_input("Base Price (Lakhs)", value=50.0, step=1.0)

# player type dropdown
player_type = st.selectbox("Player Type", ["Batter", "Bowler", "All-rounder", "WK-Batter"])

if st.button("Predict"):

    # base input
    df = pd.DataFrame([{
        "runs": runs,
        "wickets": wickets,
        "strike_rate": strike_rate,
        "average": average,
        "base_price": base_price * 0.5
    }])

    # add type columns
    df["Type_Batter"] = 1 if player_type == "Batter" else 0
    df["Type_Bowler"] = 1 if player_type == "Bowler" else 0
    df["Type_All-rounder"] = 1 if player_type == "All-rounder" else 0
    df["Type_WK-Batter"] = 1 if player_type == "WK-Batter" else 0

    # match training columns
    model_features = scaler.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]

    # scale
    df_scaled = scaler.transform(df)

    # predict (log → actual)
    pred_log = model.predict(df_scaled)[0]
    price = np.expm1(pred_log)

    # scale correction
    price = price * 2

    # clamp to realistic range
    price = max(20, min(price, 500))

    st.success(f"Predicted Price: {round(price,2)} Lakhs ({round(price/100,2)} Cr)")
