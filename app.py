import streamlit as st
import pandas as pd
import numpy as np
import joblib
#Load the trained model, scaler, and encoder
model = joblib.load("fare_model.pkl")
print("Model type:", type(model))
scaler = joblib.load("scaler.pkl")
#Streamlit app title
st.title("💰 Ride Fare Prediction App")
st.write("Enter the following details to predict ride fare.")
#--- Input fields ---
Distance = st.number_input("Distance (in km)", value=0.0, step=0.1)
Duration = st.number_input("Duration (in min)", value=0.0, step=0.1)
Demand = st.number_input("Demand Index", value=0.0, step=0.1)
Traffic = st.selectbox("Traffic Level", [1,2,3,4,5,6,7,8,9,10])
Hour = st.selectbox("Hour of Day", [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
Weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Stormy"])
Day = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
Vehicle = st.selectbox("Vehicle Type", ["Auto", "Bike", "Car"])
#--- One-hot encode region ---
#for Weather
weather_dict = {
"Clear": [1, 0, 0],   
"Rainy": [0, 1, 0],
"Stormy": [0, 0, 1]
}
Weather_encoded = weather_dict[Weather]
#for Day
day_dict = {
"Fri": [1, 0, 0, 0, 0, 0, 0],   
"Mon": [0, 1, 0, 0, 0, 0, 0],
"Sat": [0, 0, 1, 0, 0, 0, 0],
"Sun": [0, 0, 0, 1, 0, 0, 0],
"Thu": [0, 0, 0, 0, 1, 0, 0],
"Tue": [0, 0, 0, 0, 0, 1, 0],
"Wed": [0, 0, 0, 0, 0, 0, 1]
}
Day_encoded = day_dict[Day]
#for Vehicle
vehicle_dict = {
"Auto": [1, 0, 0],   
"Bike": [0, 1, 0],
"Car": [0, 0, 1]
}
Vehicle_encoded = vehicle_dict[Vehicle]
#--- Combine all features into DataFrame ---
input_data = pd.DataFrame([[Distance, Duration, Demand, Traffic, Hour] + Weather_encoded + Day_encoded + Vehicle_encoded],
columns=['distance_km', 'duration_minutes', 'demand_index', 'traffic_level', 'hour_of_day', 'weather_condition_Clear', 'weather_condition_Rainy', 'weather_condition_Stormy', 'day_of_week_Fri', 'day_of_week_Mon', 'day_of_week_Sat', 'day_of_week_Sun', 'day_of_week_Thu', 'day_of_week_Tue', 'day_of_week_Wed', 'vehicle_type_Auto', 'vehicle_type_Bike', 'vehicle_type_Car'])
#--- Scale input ---
#input_scaled = scaler.transform(input_data)
#--- Prediction ---
if st.button("Predict Ride Fare 💵"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Ride Fare: ${prediction:,.2f}")
#Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Scikit-Learn.")


