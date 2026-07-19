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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(124, 58, 237, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(236, 72, 153, 0.14) 0%, transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(20, 184, 166, 0.10) 0%, transparent 45%),
            #0f1117;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 1.6rem 0 0.6rem 0;
    }

    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: #9aa3b5;
        font-size: 1.05rem;
        margin-top: 0;
        font-weight: 400;
    }

    /* Badges */
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.7rem;
        margin: 1rem 0 2rem 0;
        flex-wrap: wrap;
    }

    .badge {
        background: rgba(129, 140, 248, 0.12);
        color: #c4b5fd;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(129, 140, 248, 0.3);
        letter-spacing: 0.2px;
    }

    /* Real container card (via st.container(key=...)) */
    .st-key-flight_form {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
        padding: 1.8rem 1.8rem 1.2rem 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: #e5e7eb;
        margin-bottom: 1.1rem;
    }

    /* Labels */
    label, .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #b8bfcc !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Selectbox / number input styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    div[data-baseweb="select"] > div:hover {
        border: 1px solid rgba(167, 139, 250, 0.5) !important;
    }

    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e5e7eb !important;
    }

    /* Slider accent */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #a78bfa !important;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 0;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.02rem;
        width: 100%;
        margin-top: 0.6rem;
        transition: all 0.35s ease;
        box-shadow: 0 4px 18px rgba(139, 92, 246, 0.35);
    }

    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.5);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #4f46e5, #9333ea, #db2777);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.4rem;
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.35);
        animation: fadeIn 0.5s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }

    .result-price {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 1.8rem;
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
    <span class="badge">⚡ XGBoost Model</span>
    <span class="badge">🎯 R² Score: 0.98</span>
    <span class="badge">📊 300K+ Flights Trained</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Input Form (real container this time)
# ─────────────────────────────────────────
with st.container(key="flight_form"):
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

    predict_clicked = st.button("Predict Price")

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
# Result
# ─────────────────────────────────────────
if predict_clicked:
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

  
