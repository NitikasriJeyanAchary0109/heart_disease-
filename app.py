"""
Heart Disease Prediction App (Framingham Dataset)
================================================
Predicts 10-year risk of Coronary Heart Disease (CHD)
using a Logistic Regression model trained on the
Framingham Heart Study dataset.
"""

import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #ff6b6b, #ee5a5a);
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
.high-risk {
    background-color: #ff4b4b;
    color: white;
}
.low-risk {
    background-color: #28a745;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model (joblib)
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("disease.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# --------------------------------------------------
# Sidebar inputs (EXACTLY 14 FEATURES)
# --------------------------------------------------
def get_user_input():
    st.sidebar.header("👤 Patient Details")

    male = st.sidebar.selectbox("Sex", ["Female", "Male"])
    male = 1 if male == "Male" else 0

    age = st.sidebar.number_input("Age", 20, 100, 50)

    currentSmoker = st.sidebar.selectbox("Current Smoker", ["No", "Yes"])
    currentSmoker = 1 if currentSmoker == "Yes" else 0

    cigsPerDay = st.sidebar.number_input(
        "Cigarettes per Day",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    BPMeds = st.sidebar.selectbox("On BP Medication", ["No", "Yes"])
    BPMeds = 1 if BPMeds == "Yes" else 0

    prevalentStroke = st.sidebar.selectbox("Previous Stroke", ["No", "Yes"])
    prevalentStroke = 1 if prevalentStroke == "Yes" else 0

    prevalentHyp = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    prevalentHyp = 1 if prevalentHyp == "Yes" else 0

    diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    diabetes = 1 if diabetes == "Yes" else 0

    totChol = st.sidebar.number_input("Total Cholesterol", 100.0, 600.0, 200.0)
    sysBP = st.sidebar.number_input("Systolic BP", 90.0, 250.0, 120.0)
    diaBP = st.sidebar.number_input("Diastolic BP", 60.0, 150.0, 80.0)

    BMI = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
    heartRate = st.sidebar.number_input("Heart Rate", 40.0, 200.0, 75.0)
    glucose = st.sidebar.number_input("Glucose", 40.0, 400.0, 80.0)

    # 🔥 EXACT FEATURE ORDER USED DURING TRAINING (14)
    features = np.array([
        male,
        age,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ])

    return features

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(model, features):
    prediction = model.predict(features.reshape(1, -1))[0]
    probability = model.predict_proba(features.reshape(1, -1))[0]
    return prediction, probability

# --------------------------------------------------
# Display results
# --------------------------------------------------
def display_results(prediction, probability):
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown(
            "<div class='prediction-box high-risk'>⚠️ High Risk of Heart Disease (10 Years)</div>",
            unsafe_allow_html=True
        )
        st.metric("Risk Probability", f"{probability[1]*100:.1f}%")
    else:
        st.markdown(
            "<div class='prediction-box low-risk'>✅ Low Risk of Heart Disease</div>",
            unsafe_allow_html=True
        )
        st.metric("Safety Probability", f"{probability[0]*100:.1f}%")

    st.warning(
        "⚠️ This prediction is for educational purposes only and "
        "must not be used as a medical diagnosis."
    )

# --------------------------------------------------
# Main app
# --------------------------------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction App</h1>
        <p>Framingham Heart Study • Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        return

    features = get_user_input()

    st.subheader("📋 Feature Vector Sent to Model")
    st.write(features)

    if st.button("🔮 Predict 10-Year CHD Risk"):
        with st.spinner("Analyzing patient data..."):
            prediction, probability = predict(model, features)
            display_results(prediction, probability)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Scikit-learn")

# --------------------------------------------------
if __name__ == "__main__":
    main()