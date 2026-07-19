import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Airline Ticket Price Predictor",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Custom CSS — Boarding Pass / Departure Board theme
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background-color: #F6F8FB;
    }

    /* ── Hero ── */
    .eyebrow {
        text-align: center;
        color: #E8A33D;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 2.2px;
        margin-bottom: 0.6rem;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.3rem;
        color: #0B2545;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .hero h1 span {
        color: #E8A33D;
    }

    .hero p {
        text-align: center;
        color: #5B6472;
        font-size: 1rem;
        max-width: 480px;
        margin: 0 auto 1.4rem auto;
        line-height: 1.5;
    }

    /* ── Trust tags ── */
    .tag-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    .tag {
        background: #ffffff;
        color: #0B2545;
        border: 1px solid #DCE2EC;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .tag::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #E8A33D;
        display: inline-block;
    }

    /* ── Boarding pass card ── */
    .st-key-boarding_pass {
        background: #ffffff;
        border: 1px solid #E4E9F0;
        border-radius: 18px;
        padding: 1.7rem 1.9rem 1.5rem 1.9rem;
        box-shadow: 0 12px 28px rgba(11, 37, 69, 0.06);
    }

    .pass-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.3rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid #EEF1F6;
    }

    .pass-header .label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        color: #0B2545;
        letter-spacing: 1.5px;
    }

    .pass-header .icon {
        font-size: 1.1rem;
    }

    /* Field labels (widget labels) */
    label, .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #5B6472 !important;
        font-weight: 700 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    div[data-baseweb="select"] > div {
        background-color: #FBFCFE !important;
        border-radius: 8px !important;
        border: 1px solid #DCE2EC !important;
    }

    div[data-baseweb="select"] > div:hover {
        border: 1px solid #E8A33D !important;
    }

    .stNumberInput input {
        background-color: #FBFCFE !important;
        border-radius: 8px !important;
        border: 1px solid #DCE2EC !important;
        color: #0B2545 !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #E8A33D !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 0 0 1px #E8A33D !important;
    }

    /* ── Perforation divider ── */
    .perforation {
        position: relative;
        border-top: 2px dashed #DCE2EC;
        margin: 1.5rem 0 1.3rem 0;
    }

    .perforation::before, .perforation::after {
        content: "";
        position: absolute;
        top: -5px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #E8A33D;
    }

    .perforation::before { left: 0; }
    .perforation::after { right: 0; }

    /* ── Button ── */
    .stButton>button {
        background: #E8A33D;
        color: #0B2545;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 0;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.25s ease;
    }

    .stButton>button:hover {
        background: #D6912E;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(232, 163, 61, 0.4);
    }

    /* ── Result stub (departure board) ── */
    .st-key-result_stub {
        background: #0B2545;
        border-radius: 18px;
        padding: 1.8rem 1.9rem;
        margin-top: 1.3rem;
        box-shadow: 0 16px 32px rgba(11, 37, 69, 0.25);
        animation: revealStub 0.5s ease;
    }

    @keyframes revealStub {
        from { opacity: 0; transform: translateY(10px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .stub-label {
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7C93B3;
        margin-bottom: 0.5rem;
    }

    .stub-price {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 2.6rem;
        color: #F2B84D;
        letter-spacing: 1px;
        margin: 0 0 0.6rem 0;
    }

    .stub-route {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #B9C6DA;
        margin-bottom: 0.9rem;
    }

    .deal-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 7px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .deal-great { background: rgba(74, 222, 128, 0.15); color: #4ade80; }
    .deal-fair  { background: rgba(242, 184, 77, 0.15); color: #F2B84D; }
    .deal-premium { background: rgba(248, 113, 113, 0.15); color: #f87171; }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        color: #99A3B5;
        font-size: 0.78rem;
        margin-top: 1.6rem;
        font-family: 'JetBrains Mono', monospace;
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
    price_stats = joblib.load('price_stats.pkl')
    return model, scaler, feature_columns, options, price_stats

model, scaler, feature_columns, options, price_stats = load_assets()

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="eyebrow">FARE PREDICTION ENGINE</div>
<div class="hero">
    <h1>Know Your <span>Fare</span> Before You Book</h1>
    <p>Instant Indian domestic airfare estimates from an XGBoost model trained on 300,000+ real flight bookings.</p>
</div>
<div class="tag-row">
    <span class="tag">XGBoost Model</span>
    <span class="tag">R² Score 0.98</span>
    <span class="tag">300K+ Flights</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Boarding Pass Input Card
# ─────────────────────────────────────────
with st.container(key="boarding_pass"):
    st.markdown("""
    <div class="pass-header">
        <span class="label">BOARDING PASS · FLIGHT DETAILS</span>
        <span class="icon">🎫</span>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown('<div class="perforation"></div>', unsafe_allow_html=True)

    predict_clicked = st.button("Estimate My Fare  →")

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

def get_deal_badge(price, travel_class):
    stats = price_stats.get(travel_class)
    if not stats:
        return None
    if price <= stats['q25']:
        return ("Great Deal", "deal-great")
    elif price <= stats['q75']:
        return ("Fair Price", "deal-fair")
    else:
        return ("Premium Fare", "deal-premium")

# ─────────────────────────────────────────
# Result Stub
# ─────────────────────────────────────────
if predict_clicked:
    with st.spinner("Calculating..."):
        price = predict_price(airline, source_city, destination_city, departure_time,
                               arrival_time, stops, travel_class, duration, days_left)

    badge = get_deal_badge(price, travel_class)
    badge_html = f'<span class="deal-badge {badge[1]}">{badge[0]}</span>' if badge else ""

    stops_label = {"zero": "Non-stop", "one": "1 Stop", "two_or_more": "2+ Stops"}.get(stops, stops)

    with st.container(key="result_stub"):
        st.markdown(f"""
        <div class="stub-label">Estimated Fare</div>
        <p class="stub-price">₹ {price:,.0f}</p>
        <div class="stub-route">{source_city.upper()[:3]} → {destination_city.upper()[:3]}  ·  {travel_class}  ·  {stops_label}  ·  {days_left}d out</div>
        {badge_html}
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">MODEL: XGBOOST · R² 0.98 · 300,000+ FLIGHTS</div>
""", unsafe_allow_html=True)

  
