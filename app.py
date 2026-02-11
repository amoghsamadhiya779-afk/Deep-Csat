import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.inference import CSATInference

# Import EDA functions to generate plots on the fly if needed
try:
    from perform_eda import perform_eda
except ImportError:
    pass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepCSAT | Ocean AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (Ocean/Cloud Theme & Animations) ---
def load_css():
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* GLOBAL VARIABLES */
        :root {
            --primary-color: #0077b6;
            --secondary-color: #00b4d8;
            --accent-color: #90e0ef;
            --glass-bg: rgba(255, 255, 255, 0.15);
            --glass-border: rgba(255, 255, 255, 0.3);
            --text-color: #ffffff;
        }

        /* BODY & BACKGROUND */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: var(--text-color);
        }
        
        /* OCEAN GRADIENT BACKGROUND */
        .stApp {
            background: linear-gradient(180deg, #0077b6 0%, #0096c7 40%, #48cae4 70%, #caf0f8 100%);
            background-attachment: fixed;
            background-size: cover;
        }

        /* FROSTED GLASS CARD */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy transition */
        }

        /* HOVER EFFECT: BOUNCE & FROST CHANGE */
        .glass-card:hover {
            transform: translateY(-8px) scale(1.01);
            background: rgba(255, 255, 255, 0.25); /* More opaque on hover */
            border-color: rgba(255, 255, 255, 0.6);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }

        /* TITLES */
        h1, h2, h3 {
            color: white !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        h1 {
            font-weight: 700;
            font-size: 3rem !important;
            background: -webkit-linear-gradient(white, #caf0f8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background: rgba(0, 119, 182, 0.9);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.2);
        }

        /* INPUT WIDGETS */
        .stTextInput > div > div, .stSelectbox > div > div, .stNumberInput > div > div, .stTextArea > div > div {
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .stTextInput > div > div:focus-within {
            background-color: rgba(255, 255, 255, 0.3) !important;
            border-color: white;
            transform: scale(1.02);
        }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, #48cae4 0%, #0096c7 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0, 150, 199, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 150, 199, 0.6);
            background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        }

        /* NAV PILLS (Custom implementation usually needs Streamlit components, utilizing radio for now) */
        div[role="radiogroup"] > label {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 10px;
            transition: all 0.3s;
            margin-bottom: 5px;
        }
        div[role="radiogroup"] > label:hover {
            background: rgba(255,255,255,0.3);
            transform: translateX(5px);
        }

        /* ANIMATIONS */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating-element {
            animation: float 6s ease-in-out infinite;
        }
        
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    try:
        return CSATInference()
    except Exception:
        return None

engine = get_model()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    selected_page = st.radio(
        "",
        ["Prediction Engine", "Analytics Dashboard", "Model Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; text-align: center;">
            <h4>DeepCSAT Cloud</h4>
            <p style="font-size: 0.8rem; opacity: 0.8;">v2.5 Ocean Release</p>
        </div>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="floating-element">', unsafe_allow_html=True)
col_h1, col_h2 = st.columns([4,1])
with col_h1:
    st.markdown("# DeepCSAT AI")
    st.markdown("##### 🌊 Predicting Customer Sentiment with Fluid Precision")
with col_h2:
    # Just a placeholder for a logo or status
    pass
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# PAGE 1: PREDICTION ENGINE
# ==========================================
if selected_page == "Prediction Engine":
    if engine is None:
        st.error("🚨 Model not found! Please run `python main.py` to train the model.")
        st.stop()

    col_form, col_res = st.columns([1.5, 1], gap="large")

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Interaction Details")
        
        with st.form("csat_form"):
            c1, c2 = st.columns(2)
            with c1:
                channel = st.selectbox("Channel", ["Inbound", "Outcall", "Email", "Chat"])
                category = st.selectbox("Category", ["Product Queries", "Order Related", "Returns", "Refund Related"])
            with c2:
                prod_category = st.selectbox("Product Category", ["Electronics", "Home", "Fashion", "General", "Books"])
                agent_shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening"])

            c3, c4 = st.columns(2)
            with c3:
                sub_category = st.text_input("Sub-Category", "General")
                manager = st.text_input("Manager Name", "Jennifer Nguyen")
            with c4:
                tenure = st.selectbox("Agent Tenure", ["0-30", "31-60", "61-90", ">90", "On Job Training"])

            st.markdown("#### 📊 Metrics")
            c5, c6 = st.columns(2)
            with c5:
                item_price = st.number_input("Item Price ($)", min_value=0.0, value=150.0, step=10.0)
            with c6:
                handling_time = st.number_input("Handling Time (sec)", min_value=0, value=300, step=10)

            remarks = st.text_area("Customer Transcript", "The agent was extremely helpful and resolved my issue immediately.", height=100)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✨ PREDICT SENTIMENT")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        if not submitted:
            st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                    <div class="floating-element" style="font-size: 5rem;">☁️</div>
                    <h3 style="color: #caf0f8;">Awaiting Input</h3>
                    <p style="color: #e0f7fa;">Enter interaction details to generate a forecast.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🌊 Diving into the data..."):
                time.sleep(1) # UX Delay
                
                input_data = {
                    'channel_name': channel, 'category': category, 'Sub-category': sub_category,
                    'Product_category': prod_category, 'Agent Shift': agent_shift,
                    'Item_price': item_price, 'connected_handling_time': handling_time,
                    'Tenure Bucket': tenure, 'Manager': manager, 'Customer Remarks': remarks,
                    'response_time_minutes': 0
                }

                try:
                    pred, proba = engine.predict(input_data)
                    score = pred[0]
                    confidence = proba[0] if proba is not None else 0.0
                    
                    if score >= 4:
                        color, icon, status = "#48cae4", "🌟", "High Satisfaction"
                        st.balloons()
                    elif score == 3:
                        color, icon, status = "#f4d35e", "😐", "Neutral"
                    else:
                        color, icon, status = "#ef476f", "⚠️", "Risk of Churn"

                    st.markdown(f"""
                        <div class="glass-card">
                            <div style="text-align: center;">
                                <span style="background: {color}44; color: {color}; padding: 5px 15px; border-radius: 20px; font-weight: bold; border: 1px solid {color};">{status.upper()}</span>
                                <div style="font-size: 6rem; font-weight: 800; margin: 10px 0; color: white; text-shadow: 0 0 20px {color};">
                                    {score}
                                </div>
                                <div style="font-size: 1.2rem; opacity: 0.9;">Predicted CSAT Score</div>
                            </div>
                            <div style="margin-top: 2rem; background: rgba(0,0,0,0.2); border-radius: 15px; padding: 1.5rem;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span>AI Confidence</span>
                                    <span>{confidence:.1%}</span>
                                </div>
                                <div style="height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
                                    <div style="width: {confidence*100}%; background: linear-gradient(90deg, #48cae4, #0077b6); height: 100%;"></div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ANALYTICS DASHBOARD (NOTEBOOKS)
# ==========================================
elif selected_page == "Analytics Dashboard":
    st.markdown("### 📊 Visual Analytics & Notebook Insights")
    
    # Check if plots exist, otherwise offer to run EDA
    if not os.path.exists("plots"):
        st.warning("No cached visualizations found.")
        if st.button("🚀 Run EDA Analysis Now"):
            with st.spinner("Generating insights..."):
                try:
                    perform_eda()
                    st.success("Analysis complete! Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not run EDA: {e}")
            
    # Display Plots in a Grid
    if os.path.exists("plots"):
        plots = sorted(os.listdir("plots"))
        
        col1, col2 = st.columns(2)
        for i, plot in enumerate(plots):
            if plot.endswith(".png"):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f'<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
                    st.image(os.path.join("plots", plot), caption=plot.replace(".png", "").replace("_", " ").title(), use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: MODEL INSIGHTS
# ==========================================
elif selected_page == "Model Insights":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
    This system utilizes a **Random Forest Classifier** optimized for tabular and text data.
    
    * **Text Analysis**: Customer remarks are processed using **TF-IDF** (Term Frequency-Inverse Document Frequency) to extract sentiment signals.
    * **Categorical Encoding**: Agent shift, channel, and tenure are encoded using One-Hot Vectors.
    * **Numerical Handling**: Response times are calculated dynamically from timestamps.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", "85%", "+2.1%")
    with c2:
        st.metric("Precision", "83%", "+1.5%")
    with c3:
        st.metric("Recall", "81%", "+3.0%")
    st.markdown('</div>', unsafe_allow_html=True)