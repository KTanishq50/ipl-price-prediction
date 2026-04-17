import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# load model and scaler
model = joblib.load("player_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# load dataset
data = pd.read_csv("playerperformance.csv")

# clean player names
data["Player_Name"] = data["Player_Name"].str.lower().str.strip()

# replace invalid values
data = data.replace("No stats", np.nan)

# convert columns to numeric
cols = ["Runs_Scored", "Wickets_Taken", "Batting_Strike_Rate", "Batting_Average"]
for col in cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

st.title("IPL Auction Price Predictor")

# input fields
runs = st.number_input("Runs", value=300.0, step=1.0)
wickets = st.number_input("Wickets", value=5.0, step=1.0)
strike_rate = st.number_input("Strike Rate", value=130.0, step=0.1)
average = st.number_input("Average", value=30.0, step=0.1)
base_price = st.number_input("Base Price (Lakhs)", value=50.0, step=1.0)

# player type input
player_type = st.selectbox("Player Type", ["Batter", "Bowler", "All-rounder", "WK-Batter"])

# prediction
if st.button("Predict"):

    # create input dataframe
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

    # scale input
    df_scaled = scaler.transform(df)

    # predict
    pred_log = model.predict(df_scaled)[0]
    price = np.expm1(pred_log)

    # adjust and clamp
    price = price * 2
    price = max(20, min(price, 500))

    st.success(f"Predicted Price: {round(price,2)} Lakhs ({round(price/100,2)} Cr)")

# separator
st.markdown("---")
st.header("Player Analysis")

# search input
search_name = st.text_input("Enter Player Name")

if search_name:

    # filter player
    player = search_name.lower().strip()
    player_data = data[data["Player_Name"] == player]

    if len(player_data) == 0:
        st.warning("Player not found in dataset")
    else:
        st.success("Player found")

        # sort by year
        player_data = player_data.sort_values("Year")
        latest = player_data.iloc[-1]

        # latest stats
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

        avg_runs = player_data["Runs_Scored"].mean(skipna=True)
        max_runs = player_data["Runs_Scored"].max(skipna=True)
        avg_wickets = player_data["Wickets_Taken"].mean(skipna=True)

        col1.metric("Avg Runs", round(avg_runs, 2) if pd.notna(avg_runs) else 0)
        col2.metric("Max Runs", int(max_runs) if pd.notna(max_runs) else 0)
        col3.metric("Avg Wickets", round(avg_wickets, 2) if pd.notna(avg_wickets) else 0)

        # runs graph
        st.subheader("Runs Over Years")

        fig, ax = plt.subplots()
        ax.plot(player_data["Year"], player_data["Runs_Scored"], marker='o')
        ax.set_xlabel("Year")
        ax.set_ylabel("Runs")
        ax.set_title("Runs vs Year")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticks(player_data["Year"][::2])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # wickets graph
        st.subheader("Wickets Over Years")

        fig, ax = plt.subplots()
        ax.plot(player_data["Year"], player_data["Wickets_Taken"], marker='o')
        ax.set_xlabel("Year")
        ax.set_ylabel("Wickets")
        ax.set_title("Wickets vs Year")
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticks(player_data["Year"][::2])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # runs histogram
        st.subheader("Runs Distribution")

        runs_data = player_data["Runs_Scored"].dropna()

        if runs_data.nunique() <= 1:
            st.info("Not enough variation in runs data")
        else:
            fig, ax = plt.subplots()
            ax.hist(runs_data, bins=5)
            ax.set_xlabel("Runs (per season)")
            ax.set_ylabel("Frequency (Years)")
            ax.set_title("Runs Distribution")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # wickets histogram
        st.subheader("Wickets Distribution")

        wickets_data = player_data["Wickets_Taken"].dropna()

        if wickets_data.nunique() <= 1:
            st.info("Not enough variation in wicket data")
        else:
            fig, ax = plt.subplots()
            wickets_data = wickets_data.astype(int)
            bins = range(int(wickets_data.min()), int(wickets_data.max()) + 2)
            ax.hist(wickets_data, bins=bins, align='left')
            ax.set_xlabel("Wickets (per season)")
            ax.set_ylabel("Frequency (Years)")
            ax.set_title("Wickets Distribution")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # career summary
        st.subheader("Career Summary")

        st.write({
            "Total Runs": int(player_data["Runs_Scored"].sum(skipna=True)),
            "Total Wickets": int(player_data["Wickets_Taken"].sum(skipna=True)),
            "Years Played": int(player_data["Year"].nunique())
        })

        # insights
        st.subheader("Insights")

        if avg_runs > avg_wickets * 20:
            st.write("Batting dominant player")
        elif avg_wickets > 10:
            st.write("Bowling dominant player")
        else:
            st.write("Balanced player")
