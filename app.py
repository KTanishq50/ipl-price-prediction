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

# player nalysis

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
            "Runs": latest["Runs_Scored"],
            "Wickets": latest["Wickets_Taken"],
            "Strike Rate": latest["Batting_Strike_Rate"],
            "Average": latest["Batting_Average"]
        })

        # metrics
        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Runs", round(player_data["Runs_Scored"].mean(), 2))
        col2.metric("Max Runs", int(player_data["Runs_Scored"].max()))
        col3.metric("Avg Wickets", round(player_data["Wickets_Taken"].mean(), 2))

        # line charts
        st.subheader("Runs Over Years")
        st.line_chart(player_data.set_index("Year")["Runs_Scored"])

        st.subheader("Wickets Over Years")
        st.line_chart(player_data.set_index("Year")["Wickets_Taken"])

        # histogram for runs
        st.subheader("Runs Distribution")
        runs_hist = np.histogram(player_data["Runs_Scored"], bins=5)
        st.bar_chart(runs_hist[0])

        # histogram for wickets
        st.subheader("Wickets Distribution")
        wickets_hist = np.histogram(player_data["Wickets_Taken"], bins=5)
        st.bar_chart(wickets_hist[0])

        # career summary
        st.subheader("Career Summary")
        st.write({
            "Total Runs": int(player_data["Runs_Scored"].sum()),
            "Total Wickets": int(player_data["Wickets_Taken"].sum()),
            "Years Played": player_data["Year"].nunique()
        })

        # insights
        st.subheader("Insights")

        avg_runs = player_data["Runs_Scored"].mean()
        avg_wickets = player_data["Wickets_Taken"].mean()

        if avg_runs > avg_wickets * 20:
            st.write("Batting dominant player")
        elif avg_wickets > 10:
            st.write("Bowling dominant player")
        else:
            st.write("Balanced player")
