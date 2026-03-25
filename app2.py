import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = sns.load_dataset("tips")

le = LabelEncoder()
for col in ["sex", "smoker", "day", "time"]:
    df[col] = le.fit_transform(df[col])

st.title("Tips Prediction")
st.header("Dataset Overview")
st.write(df)

bill = float(st.text_input("Enter total bill", "0"))
people = st.slider("People", 1, 10, 1)
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

day_map = {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}
time_map = {"Lunch": 0, "Dinner": 1}

X = df.drop("tip", axis=1)
y = df["tip"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

algo = st.sidebar.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Decision Tree"])

if algo == "Linear Regression":
    model = LinearRegression()
elif algo == "Random Forest":
    model = RandomForestRegressor()
else:
    model = DecisionTreeRegressor()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MSE: {mse:.2f}")

if st.button("Predict Tip"):
    input_data = [[bill, 1, 0, day_map[day], time_map[time], people]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Tip: {prediction[0]:.2f}")
