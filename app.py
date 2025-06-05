""" import streamlit as st
import pandas as pd
import pickle

with open('model.bin', 'rb') as f:
    model = pickle.load(f)

expected_features = [
    'age', 'gender', 'location', 'preferred_language', 'device_type',
    'total_revenue', 'auto_renew', 'plan_changes', 'months_using',
    'completion_rate', 'peak_hour_streaming', 'avg_genre_diversity',
    'max_genre', 'tenure_months', 'engagement_score', 'is_stable_plan',
    'subscription_month', 'subscription_year', 'monthly_avg_revenue'
]

label_maps = {
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'location': {'India': 0, 'US': 1, 'UK': 2, 'Other': 3},
    'preferred_language': {'English': 0, 'Tamil': 1, 'Hindi': 2, 'Other': 3},
    'device_type': {'Mobile': 0, 'TV': 1, 'Web': 2, 'Tablet': 3},
    'max_genre': {'Drama': 0, 'Action': 1, 'Comedy': 2, 'Romance': 3, 'Other': 4}
}

st.title("📊 Subscription Churn Predictor")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", list(label_maps['gender'].keys()))
location = st.selectbox("Location", list(label_maps['location'].keys()))
preferred_language = st.selectbox("Preferred Language", list(label_maps['preferred_language'].keys()))
device_type = st.selectbox("Device Type", list(label_maps['device_type'].keys()))

total_revenue = st.number_input("Total Revenue ($)", value=1000.0)
auto_renew = st.selectbox("Auto Renew Enabled?", [0, 1])
plan_changes = st.number_input("Number of Plan Changes", value=1)
months_using = st.number_input("Months Using", value=12)

completion_rate = st.slider("Completion Rate", 0.0, 1.0, 0.75)
peak_hour_streaming = st.number_input("Peak Hour Streaming", value=20)
avg_genre_diversity = st.slider("Average Genre Diversity", 0.0, 1.0, 0.6)
max_genre = st.selectbox("Most Watched Genre", list(label_maps['max_genre'].keys()))

tenure_months = st.number_input("Tenure (Months)", value=18)
engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.7)
is_stable_plan = st.selectbox("Is on Stable Plan?", [0, 1])
subscription_month = st.number_input("Subscription Month", min_value=1, max_value=12, value=5)
subscription_year = st.number_input("Subscription Year", min_value=2000, max_value=2030, value=2024)
monthly_avg_revenue = st.number_input("Monthly Average Revenue ($)", value=80.0)

input_data = {
    'age': age,
    'gender': label_maps['gender'][gender],
    'location': label_maps['location'][location],
    'preferred_language': label_maps['preferred_language'][preferred_language],
    'device_type': label_maps['device_type'][device_type],
    'total_revenue': total_revenue,
    'auto_renew': auto_renew,
    'plan_changes': plan_changes,
    'months_using': months_using,
    'completion_rate': completion_rate,
    'peak_hour_streaming': peak_hour_streaming,
    'avg_genre_diversity': avg_genre_diversity,
    'max_genre': label_maps['max_genre'][max_genre],
    'tenure_months': tenure_months,
    'engagement_score': engagement_score,
    'is_stable_plan': is_stable_plan,
    'subscription_month': subscription_month,
    'subscription_year': subscription_year,
    'monthly_avg_revenue': monthly_avg_revenue
}

input_df = pd.DataFrame([input_data])
input_df = input_df[expected_features]  

st.subheader("🔍 Encoded Input Preview")
st.dataframe(input_df)

if st.button("📈 Predict Churn Probability"):
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"🧠 Predicted Churn Probability: **{prob:.2%}**")
 """

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time # For simulation of loading

# --- Configuration for Styling ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊", # Professional icon retained
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, dark theme with enhanced styling
st.markdown("""
<style>
    /* Overall App Background and Font */
    .stApp {
        background-color: #0A0D14; /* Deep Dark Blue-Black Background */
        color: #F0F0F0; /* Light Grey for general text */
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Streamlit Containers - for main content blocks */
    .st-emotion-cache-z5fcl4 { /* Targets the main content wrapper */
        padding-top: 3rem; /* More space at the top */
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Primary Headers (H1, H2, H3) - Professional Blue for titles */
    h1, h2, h3 {
        color: #5E8BE0; /* Richer Professional Blue */
        font-weight: 700; /* Bolder */
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6); /* More prominent shadow for depth */
        margin-bottom: 0.75em;
    }
    h1 { font-size: 2.8em; }
    h2 { font-size: 2.2em; }
    h3 { font-size: 1.8em; }

    /* General Paragraph Text */
    p {
        color: #C0C0C0; /* Mid-grey for general paragraphs */
        line-height: 1.6;
    }

    /* Sidebar Styling */
    .st-emotion-cache-vk337j { /* Targets the sidebar container */
        background-color: #1E2433; /* Darker, warm blue sidebar background */
        color: #F0F0F0; /* Lighter text for sidebar */
        padding: 30px 20px; /* More padding */
        border-right: 3px solid #2F384C; /* Stronger subtle border */
        box-shadow: 4px 0px 15px rgba(0,0,0,0.6); /* Deeper shadow for depth */
    }

    /* Sidebar Header (App Logo/Title) */
    .sidebar-logo {
        font-size: 2.5rem; /* Larger font */
        font-weight: 900; /* Extra bold */
        color: #5E8BE0; /* Blue for sidebar title */
        text-align: center;
        margin-bottom: 25px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        letter-spacing: 1px; /* Subtle letter spacing */
    }
    .sidebar-slogan {
        font-size: 0.9em;
        color: #A0A0A0;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 30px;
    }

    /* Navigation Radio Buttons in Sidebar */
    .st-emotion-cache-1q1l43p { /* Target for radio button container */
        background-color: #2F384C; /* Darker background for active radio */
        border-radius: 10px; /* More rounded corners */
        padding: 12px 20px; /* Increased padding */
        margin-bottom: 12px;
        transition: background-color 0.3s ease, transform 0.2s ease; /* Smooth transition */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3); /* Subtle shadow */
    }
    .st-emotion-cache-1q1l43p:hover {
        background-color: #3A455D; /* Lighter on hover */
        transform: translateY(-2px); /* Slight lift on hover */
        box-shadow: 4px 4px 12px rgba(0,0,0,0.4);
    }
    .st-emotion-cache-1q1l43p label { /* Label inside radio button */
        color: #F0F0F0 !important; /* White text for radio labels */
        font-weight: 600;
        font-size: 1.1em;
        display: flex; /* Allow icon and text to align */
        align-items: center;
        gap: 10px; /* Space between icon and text */
    }
    .st-emotion-cache-1q1l43p .st-emotion-cache-10o5u3h { /* Radio button circle */
        border-color: #5E8BE0 !important; /* Blue circle */
        background-color: #2F384C; /* Match container background */
    }
    .st-emotion-cache-1q1l43p .st-emotion-cache-10o5u3h div { /* Radio button selected dot */
        background-color: #5E8BE0 !important; /* Blue dot */
    }

    /* Input Labels - IMPORTANT FOR VISIBILITY */
    div[data-testid="stForm"] label p,
    div[data-testid="stVerticalBlock"] label p,
    div[data-testid="stHorizontalBlock"] label p,
    label p {
        color: #F0F0F0 !important; /* Force light text for all labels */
        font-weight: bold;
        font-size: 1.05rem;
        margin-bottom: 8px; /* Add a little space below labels */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #5E8BE0; /* Professional blue button */
        color: white;
        border-radius: 10px; /* More rounded corners */
        border: none;
        padding: 14px 30px; /* Larger padding */
        font-size: 1.2rem; /* Larger font */
        font-weight: bold;
        transition: all 0.3s ease; /* Smooth transition */
        box-shadow: 4px 4px 12px rgba(0,0,0,0.5); /* Deeper shadow */
        margin-top: 20px; /* Space above buttons */
        display: flex; /* For icon alignment */
        align-items: center;
        gap: 10px;
        justify-content: center;
    }
    .stButton>button:hover {
        background-color: #4C7CDA; /* Darker blue on hover */
        transform: translateY(-4px); /* More pronounced lift */
        box-shadow: 8px 8px 20px rgba(0,0,0,0.6); /* Stronger shadow on hover */
    }
    .stButton>button:active {
        transform: translateY(0); /* Return to original position on click */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }

    /* Input fields (text, number, selectbox, slider) */
    .st-emotion-cache-1c7y2kd { /* Targets the actual input field box */
        background-color: #2F384C; /* Darker input background */
        color: #F0F0F0; /* Input text color */
        border: 1px solid #4A566C; /* Subtle border */
        border-radius: 10px; /* Rounded corners */
        padding: 0.75rem 1.25rem; /* More padding */
        box-shadow: inset 1px 1px 4px rgba(0,0,0,0.4); /* Inner shadow for depth */
    }
    /* For selectbox dropdown arrow */
    .st-emotion-cache-un8qtl {
        color: #F0F0F0; /* White arrow */
    }
    /* For selectbox options in dropdown */
    .st-emotion-cache-1l24765 { /* Target for individual options */
        background-color: #2F384C; /* Dark background for options */
        color: #F0F0F0; /* White text for options */
    }
    .st-emotion-cache-1l24765:hover {
        background-color: #4A566C; /* Lighter on hover */
    }

    /* Slider track */
    .st-emotion-cache-1v0bbps { /* Slider track background */
        background-color: #4A566C;
        border-radius: 6px;
        height: 8px; /* Thicker track */
    }
    /* Slider thumb */
    .st-emotion-cache-1c1w610 { /* Slider thumb */
        background-color: #5E8BE0; /* Blue thumb */
        border: 3px solid #4C7CDA; /* Stronger border */
        height: 20px; /* Larger thumb */
        width: 20px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
    }

    /* Success/Warning/Error messages */
    .st-emotion-cache-ch5fef, /* Info box */
    .st-emotion-cache-s8f8x3, /* Success box */
    .st-emotion-cache-1f8k0u0, /* Warning box */
    .st-emotion-cache-1y4v88n { /* Error box */
        background-color: #1E2433; /* Use sidebar background for messages */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .st-emotion-cache-ch5fef { border-left: 8px solid #5E8BE0; color: #5E8BE0; } /* Info: Blue */
    .st-emotion-cache-s8f8x3 { border-left: 8px solid #4CAF50; color: #4CAF50; } /* Success: Green */
    .st-emotion-cache-1f8k0u0 { border-left: 8px solid #FFC107; color: #FFC107; } /* Warning: Amber */
    .st-emotion-cache-1y4v88n { border-left: 8px solid #F44336; color: #F44336; } /* Error: Red */

    /* Metrics (for churn probability display) */
    .st-emotion-cache-f1x2el { /* Metric label */
        color: #A0A0A0;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .st-emotion-cache-13ln4jo { /* Metric value */
        color: #5E8BE0; /* Blue for value */
        font-size: 3em; /* Larger font */
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .st-emotion-cache-13ln4jo div { /* Metric value with arrow (delta) */
        color: inherit; /* Inherit color from parent */
        font-size: 1.2em; /* Keep it relatively sized */
    }
    .st-emotion-cache-q8sbtg { /* Metric container */
        background-color: #1E2433; /* Sidebar background */
        border-radius: 15px; /* More rounded */
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); /* Deeper shadow */
        text-align: center;
        border: 1px solid #2F384C; /* Subtle border */
    }
    /* Specific styling for the 'Likely to Churn' metric */
    .churn-likely-text { color: #F44336 !important; }
    .churn-unlikely-text { color: #4CAF50 !important; }

    /* Horizontal Rule styling */
    hr {
        border-top: 2px solid #2F384C; /* Matches darker sidebar border */
        margin-top: 40px;
        margin-bottom: 40px;
    }

    /* Custom CSS for Gauge (Placeholder - needs JavaScript for full interactivity) */
    .gauge-container {
        width: 100%;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .gauge-circle {
        position: relative;
        width: 180px;
        height: 90px; /* Half circle */
        background: conic-gradient(#4CAF50 0% var(--churn-green-end), #FFC107 var(--churn-green-end) var(--churn-yellow-end), #F44336 var(--churn-yellow-end) 100%);
        border-top-left-radius: 180px;
        border-top-right-radius: 180px;
        overflow: hidden;
        margin: auto;
        transform: rotate(180deg); /* Rotate to make it a bottom semi-circle */
    }
    .gauge-inner-circle {
        position: absolute;
        top: 10px;
        left: 10px;
        right: 10px;
        bottom: 10px;
        background-color: #0A0D14; /* App background */
        border-top-left-radius: 180px;
        border-top-right-radius: 180px;
        transform: rotate(180deg); /* Rotate back to normal */
    }
    .gauge-value {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(180deg); /* Counter-rotate value */
        font-size: 2.2em;
        font-weight: bold;
        color: white; /* Will be overridden by JS for dynamic color */
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    .gauge-label {
        margin-top: 10px;
        font-size: 1.2em;
        color: #A0A0A0;
    }

    /* Specific styles for "What If" section */
    .what-if-section {
        background-color: #1E2433;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border: 1px solid #2F384C;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_model_and_preprocessors():
    """Loads the trained model and preprocessors (label encoders, scaler)."""
    try:
        # Adjusted paths as per the training script's saving locations
        model_path = 'xgboost_general_churn_model1.pkl'
        data_path = 'churn_dataset1.csv' # Assumes this is in the same directory as app.py

        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at `{model_path}`. Please ensure it's in the correct directory.")
            st.stop()
        if not os.path.exists(data_path):
            st.error(f"Error: Dataset file not found at `{data_path}`. Please ensure it's in the correct directory.")
            st.stop()

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        dummy_df = pd.read_csv(data_path)
        
        # Preprocessing steps from training script - ensure consistency
        dummy_df = dummy_df.drop(columns=[
            'subscription_start_date',
            'subscription_end_date',
            'location',
            'user_id', # user_id also should be dropped in the original script
            'churned', # Drop target column for X_dummy
            'churn_risk_score' # Drop temporary score column
        ], errors='ignore')

        features = [
            'age', 'gender', 'preferred_language', 'device_type',
            'auto_renew', 'plan_changes', 'months_using',
            'total_revenue_inr', 'completion_rate',
            'peak_hour_streaming', 'avg_genre_diversity', 'most_watched_genre'
        ]
        
        X_dummy = dummy_df[features].copy() # Ensure we work on a copy

        # Identify categorical and numerical columns based on the original training data types
        cat_cols = X_dummy.select_dtypes(include='object').columns.tolist()
        num_cols = [col for col in X_dummy.columns if col not in cat_cols]

        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(dummy_df[col].astype(str).unique()) # Fit to all unique values from the original data
            label_encoders[col] = le

        scaler = StandardScaler()
        scaler.fit(X_dummy[num_cols]) # Fit scaler on numerical columns from the training data

        return model, label_encoders, scaler, features, cat_cols, num_cols

    except Exception as e:
        st.error(f"An error occurred while loading the model or preprocessors: {e}")
        st.stop()

model, label_encoders, scaler, features, cat_cols_trained, num_cols_trained = load_model_and_preprocessors()

# --- Prediction Function ---
def predict_churn(input_df_raw):
    """
    Makes a churn prediction based on the input data.
    Input data can be a single dictionary or a DataFrame.
    """
    input_df = input_df_raw.copy()

    # Ensure boolean 'auto_renew' is treated as integer 0/1 for consistent scaling
    if 'auto_renew' in input_df.columns:
        input_df['auto_renew'] = input_df['auto_renew'].astype(int)

    # Apply label encoding using the *fitted* encoders
    for col in cat_cols_trained:
        if col in input_df.columns:
            le = label_encoders[col]
            # Handle unseen labels by mapping them to 0 or another default
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    # Ensure the order of columns matches the training data features before scaling and prediction
    input_df = input_df[features]

    # Scale numerical features using the *fitted* scaler
    input_df[num_cols_trained] = scaler.transform(input_df[num_cols_trained])
    
    # Predict
    churn_probabilities = model.predict_proba(input_df)[:, 1]
    churn_predictions = (churn_probabilities > 0.5).astype(int) # Using 0.5 as threshold for binary classification
    
    return churn_predictions, churn_probabilities

# --- Interactive Gauge Component (using HTML/CSS/JS in Streamlit) ---
def churn_gauge(probability):
    prob_percent = probability * 100
    color_value = "green" if probability <= 0.3 else ("orange" if probability <= 0.7 else "red")
    
    # Simple CSS variable based approach for gradient
    # The conic-gradient points are in percentage, so map 0-100% to 0-100% of the half circle
    green_end = min(prob_percent, 50) # Green up to 50%
    yellow_end = min(prob_percent, 75) # Yellow up to 75%
    
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-circle" style="--churn-green-end: {green_end}%; --churn-yellow-end: {yellow_end}%;">
            <div class="gauge-inner-circle"></div>
            <div class="gauge-value" style="color: {color_value};">{prob_percent:.1f}%</div>
        </div>
        <div class="gauge-label">Churn Probability</div>
    </div>
    """, unsafe_allow_html=True)


# --- Landing Page ---
def landing_page():
    st.markdown("<h1 style='text-align: center; color: #5E8BE0; font-size: 3.5em; margin-bottom: 0;'>Welcome to</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #5E8BE0; font-size: 4.5em; margin-top: 0;'>The ChurnGuard Predictor</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
        <div style='text-align: center; padding: 30px; border-radius: 15px; background-color: #1E2433; margin-top: 40px; box-shadow: 0 6px 20px rgba(0,0,0,0.4); border: 1px solid #2F384C;'>
            <p style='font-size: 1.4em; color: #F0F0F0; line-height: 1.7;'>
                Empower your business with proactive customer retention. Our advanced machine learning model
                identifies potential customer churn before it happens, allowing you to take timely action
                and safeguard your valuable customer base.
            </p>
            <p style='font-size: 1.25em; color: #A0A0A0;'>
                Navigate to the 'Predict Churn' page using the sidebar to start predicting!
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Using a professional, relevant image
    st.image("https://images.unsplash.com/photo-1551288258-0027f6e30018?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80", use_container_width=True, caption="Understanding Customer Behavior through Data Analytics")
    
    st.markdown("""
        <div style='margin-top: 60px; text-align: center;'>
            <h3 style='color: #5E8BE0;'>Key Features:</h3>
            <ul style='list-style-type: none; padding: 0; display: inline-block; text-align: left;'>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>✔️</span> Highly accurate churn predictions</li>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>✨</span> Intuitive and user-friendly interface</li>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>💡</span> Actionable insights for retention strategies</li>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>⬇️</span> Downloadable results for reporting</li>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>🔄</span> Batch Prediction for multiple customers</li>
                <li style='margin-bottom: 15px; color: #F0F0F0;'><span style='color: #5E8BE0; font-size: 1.6em; vertical-align: middle; margin-right: 10px;'>🤔</span> "What If" Scenarios for strategic planning</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# --- Prediction Page ---
def prediction_page():
    st.markdown("<h2 style='color: #5E8BE0;'>Predict Customer Churn</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.write("Choose a prediction mode below:")
    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["Single Customer Prediction 👤", "Batch Prediction (CSV Upload) 📂"],
        horizontal=True
    )

    if prediction_mode == "Single Customer Prediction 👤":
        single_prediction_interface()
    elif prediction_mode == "Batch Prediction (CSV Upload) 📂":
        batch_prediction_interface()

def single_prediction_interface():
    st.markdown("### Enter Customer Details")
    st.write("Fill in the customer's demographic, subscription, and activity details below to assess their churn risk.")

    # Initialize session state for "What If" scenario
    if 'current_input_data' not in st.session_state:
        st.session_state.current_input_data = {}
    if 'last_prediction_made' not in st.session_state:
        st.session_state.last_prediction_made = None

    # --- Input Form ---
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Age</p>", unsafe_allow_html=True)
            age = st.number_input(" ", min_value=18, max_value=90, value=st.session_state.current_input_data.get('age', 30), help="Customer's current age.", key="age_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Gender</p>", unsafe_allow_html=True)
            gender = st.selectbox(" ", ['Male', 'Female', 'Other'], index=['Male', 'Female', 'Other'].index(st.session_state.current_input_data.get('gender', 'Male')), help="Customer's gender.", key="gender_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Preferred Language</p>", unsafe_allow_html=True)
            preferred_language = st.selectbox(" ", ['English', 'Hindi', 'Spanish', 'Tamil', 'Telugu', 'Bengali', 'Marathi'], index=['English', 'Hindi', 'Spanish', 'Tamil', 'Telugu', 'Bengali', 'Marathi'].index(st.session_state.current_input_data.get('preferred_language', 'English')), help="Primary language preference.", key="lang_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Device Type</p>", unsafe_allow_html=True)
            device_type = st.selectbox(" ", ['Smart TV', 'Mobile', 'Tablet', 'Laptop', 'Desktop'], index=['Smart TV', 'Mobile', 'Tablet', 'Laptop', 'Desktop'].index(st.session_state.current_input_data.get('device_type', 'Mobile')), help="Main device used for streaming.", key="device_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Auto Renew Subscription?</p>", unsafe_allow_html=True)
            auto_renew_val = st.session_state.current_input_data.get('auto_renew', True)
            auto_renew = st.radio(" ", [True, False], index=0 if auto_renew_val else 1, format_func=lambda x: "Yes" if x else "No", help="Is the subscription set to auto-renew?", key="renew_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Number of Plan Changes</p>", unsafe_allow_html=True)
            plan_changes = st.number_input(" ", min_value=0, max_value=10, value=st.session_state.current_input_data.get('plan_changes', 0), help="How many times the customer changed their subscription plan.", key="plan_input")
        
        with col2:
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Months Using Service</p>", unsafe_allow_html=True)
            months_using = st.number_input(" ", min_value=1, max_value=60, value=st.session_state.current_input_data.get('months_using', 12), help="Total months the customer has been subscribed.", key="months_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Total Revenue (INR)</p>", unsafe_allow_html=True)
            total_revenue_inr = st.number_input(" ", min_value=0.0, max_value=50000.0, value=st.session_state.current_input_data.get('total_revenue_inr', 1500.0), step=100.0, help="Total revenue generated from this customer in INR.", key="revenue_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Content Completion Rate</p>", unsafe_allow_html=True)
            completion_rate = st.slider(" ", min_value=0.0, max_value=1.0, value=st.session_state.current_input_data.get('completion_rate', 0.75), step=0.01, help="Average percentage of watched content completed.", key="completion_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Peak Streaming Hour</p>", unsafe_allow_html=True)
            peak_hour_streaming = st.selectbox(" ", ['Evening', 'Night', 'Afternoon', 'Morning'], index=['Evening', 'Night', 'Afternoon', 'Morning'].index(st.session_state.current_input_data.get('peak_hour_streaming', 'Evening')), help="The time of day the customer streams most.", key="peak_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Average Genre Diversity</p>", unsafe_allow_html=True)
            avg_genre_diversity = st.number_input(" ", min_value=1, max_value=10, value=st.session_state.current_input_data.get('avg_genre_diversity', 3), help="Number of distinct genres watched regularly.", key="diversity_input")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Most Watched Genre</p>", unsafe_allow_html=True)
            most_watched_genre = st.selectbox(" ", ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Documentary', 'Horror', 'Sci-Fi'], index=['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Documentary', 'Horror', 'Sci-Fi'].index(st.session_state.current_input_data.get('most_watched_genre', 'Drama')), help="The genre the customer watches most frequently.", key="genre_input")

        st.markdown("---")
        submit_button = st.form_submit_button(label="Predict Churn ▶️")

    if submit_button:
        input_data = {
            'age': age,
            'gender': gender,
            'preferred_language': preferred_language,
            'device_type': device_type,
            'auto_renew': auto_renew,
            'plan_changes': plan_changes,
            'months_using': months_using,
            'total_revenue_inr': total_revenue_inr,
            'completion_rate': completion_rate,
            'peak_hour_streaming': peak_hour_streaming,
            'avg_genre_diversity': avg_genre_diversity,
            'most_watched_genre': most_watched_genre
        }
        st.session_state.current_input_data = input_data # Save for "What If"

        with st.spinner('Calculating churn risk...'):
            time.sleep(1.5) # Simulate processing time
            churn_prediction_array, churn_probability_array = predict_churn(pd.DataFrame([input_data]))
            churn_prediction = churn_prediction_array[0]
            churn_probability = churn_probability_array[0]
            st.session_state.last_prediction_made = (churn_prediction, churn_probability)
        
        display_single_prediction_results(input_data, churn_prediction, churn_probability)

    # Display "What If" section only if a prediction has been made
    if st.session_state.last_prediction_made:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="what-if-section">
            <h3 style='color: #5E8BE0;'>What If Scenario Analysis 🤔</h3>
            <p style='color: #C0C0C0;'>Adjust parameters below to see how changes might affect the churn probability.</p>
        """, unsafe_allow_html=True)

        col_whatif_1, col_whatif_2 = st.columns(2)
        with col_whatif_1:
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Change Completion Rate</p>", unsafe_allow_html=True)
            what_if_completion_rate = st.slider(" ", min_value=0.0, max_value=1.0, value=st.session_state.current_input_data['completion_rate'], step=0.01, key="what_if_completion")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Change Plan Changes</p>", unsafe_allow_html=True)
            what_if_plan_changes = st.number_input(" ", min_value=0, max_value=10, value=st.session_state.current_input_data['plan_changes'], key="what_if_plan_changes")

        with col_whatif_2:
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Change Total Revenue (INR)</p>", unsafe_allow_html=True)
            what_if_total_revenue_inr = st.number_input(" ", min_value=0.0, max_value=50000.0, value=st.session_state.current_input_data['total_revenue_inr'], step=100.0, key="what_if_revenue")
            
            st.markdown("<p style='color:#F0F0F0; font-weight:bold;'>Change Months Using Service</p>", unsafe_allow_html=True)
            what_if_months_using = st.number_input(" ", min_value=1, max_value=60, value=st.session_state.current_input_data['months_using'], key="what_if_months")

        # Create a temporary input data for "What If" scenario
        what_if_input_data = st.session_state.current_input_data.copy()
        what_if_input_data['completion_rate'] = what_if_completion_rate
        what_if_input_data['plan_changes'] = what_if_plan_changes
        what_if_input_data['total_revenue_inr'] = what_if_total_revenue_inr
        what_if_input_data['months_using'] = what_if_months_using

        what_if_churn_prediction_array, what_if_churn_probability_array = predict_churn(pd.DataFrame([what_if_input_data]))
        what_if_churn_probability = what_if_churn_probability_array[0]

        st.markdown(f"**Updated Churn Probability:**")
        churn_gauge(what_if_churn_probability)
        if what_if_churn_probability > 0.5:
            st.markdown(f"<p style='color:#F44336; font-weight:bold; text-align:center;'>Likely to Churn ({what_if_churn_probability:.2%})</p>", unsafe_allow_html=True)
            st.info("💡 **Retention Tip:** Consider offering a personalized discount or premium content trial to re-engage this customer, especially if 'Completion Rate' is low or 'Plan Changes' is high.")
        else:
            st.markdown(f"<p style='color:#4CAF50; font-weight:bold; text-align:center;'>Unlikely to Churn ({what_if_churn_probability:.2%})</p>", unsafe_allow_html=True)
            st.success("👍 **Retention Tip:** Keep this customer engaged with new content, personalized recommendations, and loyalty rewards.")
        st.markdown("</div>", unsafe_allow_html=True) # Close what-if-section div


def display_single_prediction_results(input_data, churn_prediction, churn_probability):
    st.markdown("### Prediction Results:")
    col_pred, col_prob, col_gauge = st.columns([1, 1, 1.5]) # Adjusted column width for gauge
    with col_pred:
        if churn_prediction == 1:
            st.markdown(f"<h3 class='churn-likely-text'>Likely to Churn 🙁</h3>", unsafe_allow_html=True)
            st.metric(label="Action Required", value="High Risk", delta_color="inverse")
            st.markdown(f"<p style='color:#F44336; font-weight:bold; font-size:1.1em;'>**Potential Loss!**</p>", unsafe_allow_html=True)
            st.warning("Immediate intervention recommended. Consider a personalized offer or customer service outreach.")
            st.balloons() # Confetti for good news (or bad, depending on perspective!)
        else:
            st.markdown(f"<h3 class='churn-unlikely-text'>Unlikely to Churn 🎉</h3>", unsafe_allow_html=True)
            st.metric(label="Good Retention", value="Low Risk")
            st.markdown(f"<p style='color:#4CAF50; font-weight:bold; font-size:1.1em;'>**Customer is Stable!**</p>", unsafe_allow_html=True)
            st.success("Great! Continue providing excellent service and value.")
            st.snow() # Snow for good news

    with col_prob:
        st.metric(label="Probability", value=f"{churn_probability:.2%}")

    with col_gauge:
        churn_gauge(churn_probability)

    st.markdown("---")
    st.markdown("### Download Results")
    result_df = pd.DataFrame({
        "Feature": list(input_data.keys()),
        "Value": list(input_data.values())
    })
    result_df.loc[len(result_df)] = ["Predicted Churn (1=Yes, 0=No)", churn_prediction]
    result_df.loc[len(result_df)] = ["Churn Probability", f"{churn_probability:.2%}"]

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Prediction as CSV 💾",
        data=csv,
        file_name="churn_prediction_results.csv",
        mime="text/csv",
        help="Download the input data and prediction result as a CSV file."
    )

    st.markdown("---")
    st.markdown("### Customer Retention Strategies")
    if churn_prediction == 1:
        st.info("""
        **Recommended Actions for High Churn Risk:**
        * **Personalized Offers:** Provide a discount on their next subscription, or a free trial of premium features.
        * **Customer Service Outreach:** Proactively contact them to understand any issues or dissatisfaction.
        * **Feedback Collection:** Implement surveys to understand their reasons for potential churn.
        * **Re-engagement Campaigns:** Target them with content relevant to their past viewing habits or new releases.
        * **Feature Highlight:** Educate them on underutilized features that might add value.
        """)
    else:
        st.info("""
        **Strategies for Retaining Low Churn Risk Customers:**
        * **Loyalty Programs:** Reward long-term customers with exclusive content, discounts, or early access.
        * **Personalized Recommendations:** Continue to provide highly relevant content suggestions.
        * **Community Building:** Encourage engagement through forums, social media, or exclusive events.
        * **Feedback & Innovation:** Solicit feedback to improve the service and keep them delighted.
        * **Upselling/Cross-selling:** Introduce them to higher-tier plans or complementary services at appropriate times.
        """)

def batch_prediction_interface():
    st.markdown("### Upload CSV for Batch Prediction")
    st.write("Upload a CSV file containing customer data to get churn predictions for multiple customers.")
    st.info("❗ **Important:** Your CSV file must contain the following columns exactly as specified (case-sensitive): "
            "`age`, `gender`, `preferred_language`, `device_type`, `auto_renew`, `plan_changes`, "
            "`months_using`, `total_revenue_inr`, `completion_rate`, `peak_hour_streaming`, `avg_genre_diversity`, `most_watched_genre`.")
    st.download_button(
        label="Download Sample CSV Template 📥",
        data=pd.DataFrame(columns=features).to_csv(index=False).encode('utf-8'),
        file_name="sample_churn_data.csv",
        mime="text/csv",
        help="Download a CSV template to structure your batch prediction data."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_df.head())

            # Check if all required columns are present
            missing_cols = [col for col in features if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Error: Missing required columns in the uploaded CSV: {', '.join(missing_cols)}. "
                         f"Please ensure your CSV matches the template.")
                return

            if st.button("Run Batch Prediction ▶️"):
                with st.spinner("Processing batch predictions... This may take a moment for large files."):
                    batch_predictions, batch_probabilities = predict_churn(batch_df[features])
                    
                    batch_df['Predicted_Churn'] = batch_predictions
                    batch_df['Churn_Probability'] = batch_probabilities.round(4) # Round for display

                    st.success("Batch prediction complete!")
                    st.write("### Batch Prediction Results:")
                    st.dataframe(batch_df)

                    csv_output = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Batch Results as CSV ⬇️",
                        data=csv_output,
                        file_name="batch_churn_predictions.csv",
                        mime="text/csv",
                        help="Download the original data with added churn predictions and probabilities."
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")
            st.info("Please ensure the CSV is correctly formatted and columns match the expected types.")


# --- About Page ---
def about_page():
    st.markdown("<h2 style='color: #5E8BE0;'>About ChurnGuard Predictor</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='background-color: #1E2433; padding: 30px; border-radius: 15px; box-shadow: 0 6px 20px rgba(0,0,0,0.4); border: 1px solid #2F384C;'>
            <p style='font-size: 1.1em; line-height: 1.6; color: #C0C0C0;'>
                The <b>ChurnGuard Predictor</b> is a state-of-the-art machine learning application designed to help businesses
                proactively identify customers at high risk of churn. By leveraging an advanced
                <b>XGBoost Classifier</b>, trained on comprehensive customer data, this tool provides
                highly accurate churn probabilities and predictions.
            </p>
            <p style='font-size: 1.1em; line-height: 1.6; color: #C0C0C0;'>
                Our model considers various factors including user demographics, subscription history,
                and content consumption patterns to provide actionable insights. The goal is to
                empower businesses to implement targeted retention strategies, reduce customer
                attrition, and maximize customer lifetime value.
            </p>
            <h4 style='color: #5E8BE0; margin-top: 30px;'>Technical Details:</h4>
            <ul style='list-style-type: none; padding: 0;'>
                <li style='margin-bottom: 10px; color: #F0F0F0;'><span style='color: #5E8BE0; margin-right: 8px;'>•</span> <b>Model:</b> XGBoost Classifier</li>
                <li style='margin-bottom: 10px; color: #F0F0F0;'><span style='color: #5E8BE0; margin-right: 8px;'>•</span> <b>Data Preprocessing:</b> Label Encoding for categorical features, StandardScaler for numerical features.</li>
                <li style='margin-bottom: 10px; color: #F0F0F0;'><span style='color: #5E8BE0; margin-right: 8px;'>•</span> <b>Imbalance Handling:</b> SMOTE applied during training to address class imbalance.</li>
                <li style='margin-bottom: 10px; color: #F0F0F0;'><span style='color: #5E8BE0; margin-right: 8px;'>•</span> <b>Performance:</b> Achieved high ROC AUC and F1 scores on the test set.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
        <p style='text-align: center; color: #A0A0A0;'>
            Developed by<br>
            <span style='color: #5E8BE0; font-weight: bold; font-size: 1.1em;'>Cache Me Team</span>
        </p>
        <p style='text-align: center; color: #A0A0A0; font-size: 0.9em;'>
            Dhivyesh K | Harsith S | Aryan Ramanand
        </p>
    """, unsafe_allow_html=True)


# --- Main App Logic ---
def main():
    with st.sidebar:
        st.markdown("<div class='sidebar-logo'>ChurnGuard</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-slogan'>Your Churn Prediction Partner</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["Home 🏠", "Predict Churn 🔮", "About This App ℹ️"],
            help="Select a page to navigate through the application."
        )

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin-top: 30px;'>
                <p style='font-size: 0.9em; color: #A0A0A0;'>
                    Developed by<br>
                    <span style='color: #5E8BE0; font-weight: bold;'>Cache Me Team</span>
                </p>
                <p style='font-size: 0.8em; color: #A0A0A0;'>
                    Dhivyesh K<br>
                    Harsith S<br>
                    Aryan Ramanand
                </p>
            </div>
        """, unsafe_allow_html=True)

    if "Home" in page:
        landing_page()
    elif "Predict Churn" in page:
        prediction_page()
    elif "About This App" in page:
        about_page()

if __name__ == "__main__":
    main()
