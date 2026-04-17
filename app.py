import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("player_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# load dataset
data = pd.read_csv("playerperformance.csv")

# clean player names
data["Player_Name"] = data["Player_Name"].str.lower().str.strip()

# handle invalid values
data = data.replace("No stats", np.nan)

# convert numeric columns
cols = ["Runs_Scored", "Wickets_Taken", "Batting_Strike_Rate", "Batting_Average"]
for col in cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

st.title("IPL Auction Price Predictor")

# predictions

runs = st.number_input("Runs", value=300.0, step=1.0)
wickets = st.number_input("Wickets", value=5.0, step=1.0)
strike_rate = st.number_input("Strike Rate", value=130.0, step=0.1)
average = st.number_input("Average", value=30.0, step=0.1)
base_price = st.number_input("Base Price (Lakhs)", value=50.0, step=1.0)

player_type = st.selectbox("Player Type", ["Batter", "Bowler", "All-rounder", "WK-Batter"])

if st.button("Predict"):

    df = pd.DataFrame([{
        "runs": runs,
        "wickets": wickets,
        "strike_rate": strike_rate,
        "average": average,
        "base_price": base_price * 0.5
    }])

    df["Type_Batter"] = 1 if player_type == "Batter" else 0
    df["Type_Bowler"] = 1 if player_type == "Bowler" else 0
    df["Type_All-rounder"] = 1 if player_type == "All-rounder" else 0
    df["Type_WK-Batter"] = 1 if player_type == "WK-Batter" else 0

    model_features = scaler.feature_names_in_

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]

    df_scaled = scaler.transform(df)

    pred_log = model.predict(df_scaled)[0]
    price = np.expm1(pred_log)

    price = price * 2
    price = max(20, min(price, 500))

    st.success(f"Predicted Price: {round(price,2)} Lakhs ({round(price/100,2)} Cr)")

# player analysis

st.markdown("---")
st.header("Player Analysis")

search_name = st.text_input("Enter Player Name")

if search_name:

    player = search_name.lower().strip()
    player_data = data[data["Player_Name"] == player]

    if len(player_data) == 0:
        st.warning("Player not found in dataset")
    else:
        st.success("Player found")

        # sort by year
        player_data = player_data.sort_values("Year")

        # latest stats
        latest = player_data.iloc[-1]

        st.subheader("Latest Performance")
        st.write({
            "Runs": int(latest["Runs_Scored"]) if pd.notna(latest["Runs_Scored"]) else 0,
            "Wickets": int(latest["Wickets_Taken"]) if pd.notna(latest["Wickets_Taken"]) else 0,
            "Strike Rate": float(latest["Batting_Strike_Rate"]) if pd.notna(latest["Batting_Strike_Rate"]) else 0,
            "Average": float(latest["Batting_Average"]) if pd.notna(latest["Batting_Average"]) else 0
        })

        # metrics
        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Runs", round(player_data["Runs_Scored"].mean(skipna=True), 2))
        col2.metric("Max Runs", int(player_data["Runs_Scored"].max(skipna=True)))
        col3.metric("Avg Wickets", round(player_data["Wickets_Taken"].mean(skipna=True), 2))

        # line charts
        st.subheader("Runs Over Years")
        st.line_chart(player_data.set_index("Year")["Runs_Scored"])

        st.subheader("Wickets Over Years")
        st.line_chart(player_data.set_index("Year")["Wickets_Taken"])

        # histogram runs
        st.subheader("Runs Distribution")
        runs_hist = np.histogram(player_data["Runs_Scored"].dropna(), bins=5)
        st.bar_chart(runs_hist[0])

        # histogram wickets
        st.subheader("Wickets Distribution")
        wickets_hist = np.histogram(player_data["Wickets_Taken"].dropna(), bins=5)
        st.bar_chart(wickets_hist[0])

        # career summary
        st.subheader("Career Summary")
        st.write({
            "Total Runs": int(player_data["Runs_Scored"].sum(skipna=True)),
            "Total Wickets": int(player_data["Wickets_Taken"].sum(skipna=True)),
            "Years Played": player_data["Year"].nunique()
        })

        # insights
        st.subheader("Insights")

        avg_runs = player_data["Runs_Scored"].mean(skipna=True)
        avg_wickets = player_data["Wickets_Taken"].mean(skipna=True)

        if avg_runs > avg_wickets * 20:
            st.write("Batting dominant player")
        elif avg_wickets > 10:
            st.write("Bowling dominant player")
        else:
            st.write("Balanced player")
