import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Airline Price Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
    }

    .hero {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f4e8c, #4a90d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero p {
        color: #667085;
        font-size: 1rem;
        margin-top: 0;
    }

    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin: 0.8rem 0 1.5rem 0;
        flex-wrap: wrap;
    }

    .badge {
        background: #eef2ff;
        color: #3a4d8f;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #dbe4ff;
    }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 2px 14px rgba(30, 40, 70, 0.06);
        border: 1px solid #eef0f4;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #1f2937;
        margin-bottom: 0.8rem;
    }

    .stButton>button {
        background: linear-gradient(90deg, #1f4e8c, #4a90d9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 0;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: 0.2s;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    .result-card {
        background: linear-gradient(135deg, #1f4e8c, #4a90d9);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        margin-top: 1.2rem;
        box-shadow: 0 8px 24px rgba(31, 78, 140, 0.25);
    }

    .result-label {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-bottom: 0.3rem;
    }

    .result-price {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
    }

    .footer-note {
        text-align: center;
        color: #98a2b3;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Model Assets
# ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    options = joblib.load('dropdown_options.pkl')
    return model, scaler, feature_columns, options

model, scaler, feature_columns, options = load_assets()

# ─────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✈️ Airline Ticket Price Predictor</h1>
    <p>Get an instant fare estimate powered by machine learning</p>
</div>
<div class="badge-row">
    <span class="badge">XGBoost Model</span>
    <span class="badge">R² Score: 0.98</span>
    <span class="badge">300K+ Flights Trained</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🛫 Flight Details</div>', unsafe_allow_html=True)

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
    duration = st.number_input("Flight Duration (hrs)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)

days_left = st.slider("Days Left Until Departure", min_value=1, max_value=49, value=15)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────
def predict_price(airline, source_city, destination_city, departure_time,
                   arrival_time, stops, travel_class, duration, days_left):

    input_data = {'duration': duration, 'days_left': days_left}
    for col in feature_columns:
        if col not in ['duration', 'days_left']:
            input_data[col] = 0

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
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])

    prediction = model.predict(input_df)[0]
    return max(0, prediction)

# ─────────────────────────────────────────
# Predict Button + Result
# ─────────────────────────────────────────
if st.button("Predict Price"):
    with st.spinner("Calculating best estimate..."):
        price = predict_price(airline, source_city, destination_city, departure_time,
                               arrival_time, stops, travel_class, duration, days_left)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Ticket Price</div>
        <p class="result-price">₹ {price:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">Prediction powered by an XGBoost regression model trained on 300,000+ real flight booking records.</div>
""", unsafe_allow_html=True)

  
