import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model assets
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
options = joblib.load('dropdown_options.pkl')

# Page config
st.set_page_config(page_title="Airline Price Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Airline Ticket Price Predictor")
st.markdown("Fill in the flight details below to get an estimated ticket price.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", options['airline'])
    source_city = st.selectbox("Source City", options['source_city'])
    destination_city = st.selectbox("Destination City", options['destination_city'])
    travel_class = st.selectbox("Class", options['class'])

with col2:
    departure_time = st.selectbox("Departure Time", options['departure_time'])
    arrival_time = st.selectbox("Arrival Time", options['arrival_time'])
    stops = st.selectbox("Number of Stops", options['stops'])
    duration = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)

days_left = st.slider("Days Left Until Departure", min_value=1, max_value=49, value=15)

st.divider()

def predict_price(airline, source_city, destination_city, departure_time,
                  arrival_time, stops, travel_class, duration, days_left):

    input_data = {
        'duration': duration,
        'days_left': days_left,
    }

    # Add all one-hot encoded columns as 0 first
    for col in feature_columns:
        if col not in ['duration', 'days_left']:
            input_data[col] = 0

    # Set the relevant encoded columns to 1
    def set_encoded(prefix, value):
        col = f"{prefix}_{value}"
        if col in input_data:
            input_data[col] = 1

    set_encoded('airline', airline)
    set_encoded('source_city', source_city)
    set_encoded('destination_city', destination_city)
    set_encoded('departure_time', departure_time)
    set_encoded('arrival_time', arrival_time)
    set_encoded('stops', stops)
    set_encoded('class', travel_class)

    input_df = pd.DataFrame([input_data])[feature_columns]

    # Scale numeric features
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])

    prediction = model.predict(input_df)[0]
    return max(0, prediction)

if st.button("Predict Price", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        price = predict_price(airline, source_city, destination_city, departure_time,
                              arrival_time, stops, travel_class, duration, days_left)

    st.success(f"### Estimated Ticket Price: ₹ {price:,.0f}")

    st.markdown("---")
    st.caption("Prediction made using an XGBoost regression model trained on 300,000+ real flight booking records.")
