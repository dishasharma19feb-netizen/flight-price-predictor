import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SkyFare Price Predictor",
    page_icon="🛫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Custom CSS — Aero-Precision System
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #0047AB;
        --secondary: #00AEEF;
        --tertiary: #F8FAFC;
        --neutral: #64748B;
        --heading: #1E293B;
        --success: #16A34A;
        --warning: #D97706;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sky-Flow animated background ── */
    .stApp {
        background: linear-gradient(120deg, #0047AB, #00AEEF, #F8FAFC, #0047AB);
        background-size: 400% 400%;
        animation: skyFlow 22s ease infinite;
    }

    @keyframes skyFlow {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp { animation: none; }
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 1.6rem 0 1.2rem 0;
    }

    .hero h1 {
        font-weight: 800;
        font-size: 2.35rem;
        color: #ffffff;
        text-shadow: 0 2px 18px rgba(0, 20, 60, 0.25);
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.88);
        font-size: 1rem;
        font-weight: 500;
        text-shadow: 0 1px 12px rgba(0, 20, 60, 0.2);
    }

    /* ── Glass card base ── */
    .st-key-search_card, .st-key-result_card, .st-key-forecast_card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 1.7rem 1.9rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 12px 32px rgba(0, 30, 80, 0.14);
    }

    .card-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--heading);
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Labels & inputs ── */
    label, .stSlider label, .stNumberInput label, .stSelectbox label {
        color: var(--neutral) !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(100, 116, 139, 0.25) !important;
    }

    div[data-baseweb="select"] > div:hover {
        border: 1px solid var(--secondary) !important;
    }

    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(100, 116, 139, 0.25) !important;
        color: var(--heading) !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--secondary) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 0 0 1px var(--secondary) !important;
    }

    /* ── Button ── */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.78rem 0;
        font-weight: 700;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 6px 18px rgba(0, 71, 171, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0, 71, 171, 0.4);
    }

    /* ── Result price ── */
    .price-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--neutral);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .price-value {
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--primary);
        margin: 0 0 0.9rem 0;
    }

    /* ── Recommendation badge ── */
    .rec-badge {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .rec-buy {
        background: rgba(22, 163, 74, 0.12);
        color: var(--success);
        border: 1px solid rgba(22, 163, 74, 0.25);
    }

    .rec-wait {
        background: rgba(217, 119, 6, 0.12);
        color: var(--warning);
        border: 1px solid rgba(217, 119, 6, 0.25);
    }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.78rem;
        margin-top: 0.6rem;
        font-weight: 500;
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
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛫 SkyFare Price Predictor</h1>
    <p>Search a route to get an instant fare estimate and a 30-day price forecast.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Search Card
# ─────────────────────────────────────────
with st.container(key="search_card"):
    st.markdown('<div class="card-title">🔍 Search Flights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        source_city = st.selectbox("From", options['source_city'])
        airline = st.selectbox("Airline", options['airline'])
        departure_time = st.selectbox("Departure Time", options['departure_time'])
        stops = st.selectbox("Stops", options['stops'])
    with col2:
        destination_city = st.selectbox("To", options['destination_city'])
        travel_class = st.selectbox("Class", options['class'])
        arrival_time = st.selectbox("Arrival Time", options['arrival_time'])
        duration = st.number_input("Duration (hrs)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)

    days_left = st.slider("Days Left Until Departure", min_value=1, max_value=49, value=15)

    predict_clicked = st.button("Predict Fare")

# ─────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────
def build_input_row(days_left_val):
    input_data = {'duration': duration, 'days_left': days_left_val}
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
    return input_data

def predict_price(days_left_val):
    row = build_input_row(days_left_val)
    input_df = pd.DataFrame([row])[feature_columns]
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])
    prediction = model.predict(input_df)[0]
    return max(0, float(prediction))

def predict_forecast(max_days=30):
    days_range = list(range(1, max_days + 1))
    rows = [build_input_row(d) for d in days_range]
    input_df = pd.DataFrame(rows)[feature_columns]
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])
    preds = model.predict(input_df)
    return days_range, np.maximum(0, preds)

# ─────────────────────────────────────────
# Result + Forecast
# ─────────────────────────────────────────
if predict_clicked:
    with st.spinner("Calculating fare and 30-day forecast..."):
        price = predict_price(days_left)
        forecast_days, forecast_prices = predict_forecast(30)

    min_price = float(np.min(forecast_prices))
    min_day = forecast_days[int(np.argmin(forecast_prices))]
    is_good_time = price <= min_price * 1.03

    with st.container(key="result_card"):
        st.markdown('<div class="card-title">🎫 Your Estimated Fare</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="price-label">{source_city} → {destination_city} · {travel_class} · {days_left} days out</div>
        <p class="price-value">₹ {price:,.0f}</p>
        """, unsafe_allow_html=True)

        if is_good_time:
            st.markdown("""
            <div class="rec-badge rec-buy">✅ Buy Now — this is close to the best price in the next 30 days.</div>
            """, unsafe_allow_html=True)
        else:
            savings = price - min_price
            st.markdown(f"""
            <div class="rec-badge rec-wait">⏳ Wait — fares are typically ₹{savings:,.0f} cheaper around {min_day} days before departure.</div>
            """, unsafe_allow_html=True)

    with st.container(key="forecast_card"):
        st.markdown('<div class="card-title">📈 30-Day Price Forecast</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_days, y=forecast_prices,
            mode='lines', line=dict(color='#0047AB', width=3),
            fill='tozeroy', fillcolor='rgba(0, 174, 239, 0.12)',
            name='Forecasted Price'
        ))
        fig.add_trace(go.Scatter(
            x=[days_left], y=[price],
            mode='markers', marker=dict(color='#00AEEF', size=12, line=dict(color='white', width=2)),
            name='Selected Date'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#1E293B'),
            xaxis_title='Days Before Departure',
            yaxis_title='Price (INR)',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(gridcolor='rgba(100,116,139,0.15)'),
            yaxis=dict(gridcolor='rgba(100,116,139,0.15)'),
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("""
<div class="footer-note">Model: XGBoost · R² 0.98 · Trained on 300,000+ real flight bookings</div>
""", unsafe_allow_html=True)

  
