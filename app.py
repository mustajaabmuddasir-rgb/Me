import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

st.title("Mineral Price Prediction App")

option = st.sidebar.selectbox("Select Mineral", ["Diamond", "Gold"])
algo = st.sidebar.selectbox("Select Algorithm", ["Linear Regression", "Random Forest", "Decision Tree"])

if option == "Diamond":
    df = sns.load_dataset("diamonds")

    le_cut = LabelEncoder()
    le_color = LabelEncoder()
    le_clarity = LabelEncoder()

    df["cut"] = le_cut.fit_transform(df["cut"])
    df["color"] = le_color.fit_transform(df["color"])
    df["clarity"] = le_clarity.fit_transform(df["clarity"])

    X = df[["carat", "cut", "color", "clarity", "depth", "table"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if algo == "Linear Regression":
        model = LinearRegression()
    elif algo == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = DecisionTreeRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.write("R2 Score:", r2_score(y_test, y_pred))

    st.sidebar.subheader("Diamond Inputs")
    carat = st.sidebar.number_input("Carat")
    cut = st.sidebar.selectbox("Cut", list(le_cut.classes_))
    color = st.sidebar.selectbox("Color", list(le_color.classes_))
    clarity = st.sidebar.selectbox("Clarity", list(le_clarity.classes_))
    depth = st.sidebar.number_input("Depth")
    table = st.sidebar.number_input("Table")

    if st.sidebar.button("Predict Diamond Price"):
        cut_enc = le_cut.transform([cut])[0]
        color_enc = le_color.transform([color])[0]
        clarity_enc = le_clarity.transform([clarity])[0]

        pred = model.predict([[carat, cut_enc, color_enc, clarity_enc, depth, table]])
        st.success(f"Predicted Diamond Price: {pred[0]:.2f}")

elif option == "Gold":
    gold_url = "https://datahub.io/core/gold-prices/r/monthly.csv"
    gold_df = pd.read_csv(gold_url)

    gold_df["Date"] = pd.to_datetime(gold_df["Date"])
    gold_df["Year"] = gold_df["Date"].dt.year
    gold_df["Month"] = gold_df["Date"].dt.month
    gold_df["YearMonth"] = gold_df["Year"] + gold_df["Month"] / 12
    gold_df = gold_df.dropna()

    Xg = gold_df[["YearMonth"]]
    yg = gold_df["Price"]

    Xg_train, Xg_test, yg_train, yg_test = train_test_split(Xg, yg, test_size=0.2)

    if algo == "Linear Regression":
        model = LinearRegression()
    elif algo == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = DecisionTreeRegressor()

    model.fit(Xg_train, yg_train)

    yg_pred = model.predict(Xg_test)
    st.write("Gold R2 Score:", r2_score(yg_test, yg_pred))

    st.sidebar.subheader("Gold Inputs")
    year = st.sidebar.slider("Year", 2000, 2100, 2030)
    month = st.sidebar.selectbox(
        "Month",
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    )

    month_map = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
        "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
    }

    if st.sidebar.button("Predict Gold Price"):
        m = month_map[month]
        year_month = year + m / 12
        future_price = model.predict([[year_month]])
        st.success(f"Predicted Gold Price ({month}-{year}): {future_price[0]:.2f}")