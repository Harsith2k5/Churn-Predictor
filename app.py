import streamlit as st
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
