"""
Heart Disease Prediction App
============================
This Streamlit app predicts the likelihood of heart disease based on patient features.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .high-risk {
        background-color: #ff4b4b;
        color: white;
    }
    .low-risk {
        background-color: #28a745;
        color: white;
    }
    .feature-description {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model from pickle file"""
    try:
        with open('disease.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_user_input():
    """
    Create input fields for all heart disease features
    Returns a numpy array with user inputs
    """
    st.sidebar.header("Patient Information")
    st.sidebar.markdown("---")
    
    # Demographic Information
    st.sidebar.subheader("👤 Demographic Information")
    
    age = st.sidebar.number_input(
        "Age (years)",
        min_value=20,
        max_value=100,
        value=50,
        help="Age of the patient in years"
    )
    
    sex = st.sidebar.selectbox(
        "Sex",
        options=["Male", "Female"],
        help="Gender of the patient"
    )
    sex_encoded = 1 if sex == "Male" else 0
    
    # Clinical Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Clinical Measurements")
    
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina",
            "Atypical Angina", 
            "Non-anginal Pain",
            "Asymptomatic"
        ],
        help="Type of chest pain experienced"
    )
    cp_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    
    trestbps = st.sidebar.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120,
        help="Resting blood pressure measured in mm Hg"
    )
    
    chol = st.sidebar.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        help="Serum cholesterol level in mg/dl"
    )
    
    fbs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No", "Yes"],
        help="Whether fasting blood sugar is greater than 120 mg/dl"
    )
    fbs_encoded = 1 if fbs == "Yes" else 0
    
    restecg = st.sidebar.selectbox(
        "Resting Electrocardiographic Results",
        options=[
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ],
        help="Resting electrocardiogram results"
    )
    restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    
    thalach = st.sidebar.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=150,
        help="Maximum heart rate achieved during exercise"
    )
    
    exang = st.sidebar.selectbox(
        "Exercise Induced Angina",
        options=["No", "Yes"],
        help="Whether exercise induced angina"
    )
    exang_encoded = 1 if exang == "Yes" else 0
    
    oldpeak = st.sidebar.number_input(
        "ST Depression Induced by Exercise",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=["Upsloping", "Flat", "Downsloping"],
        help="The slope of the peak exercise ST segment"
    )
    slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
    
    ca = st.sidebar.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy",
        options=[0, 1, 2, 3],
        help="Number of major vessels colored by fluoroscopy (0-3)"
    )
    
    thal = st.sidebar.selectbox(
        "Thalassemia",
        options=[
            "Normal",
            "Fixed Defect",
            "Reversable Defect"
        ],
        help="Thalassemia type"
    )
    thal_encoded = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal)
    
    # Create feature array in the correct order
    features = np.array([
        age,
        sex_encoded,
        cp_encoded,
        trestbps,
        chol,
        fbs_encoded,
        restecg_encoded,
        thalach,
        exang_encoded,
        oldpeak,
        slope_encoded,
        ca,
        thal_encoded
    ])
    
    return features

def make_prediction(model, features):
    """Make prediction using the model"""
    try:
        prediction = model.predict(features.reshape(1, -1))
        probability = model.predict_proba(features.reshape(1, -1))
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def display_results(prediction, probability):
    """Display prediction results with visual feedback"""
    st.markdown("---")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box high-risk">
                ⚠️ High Risk of Heart Disease
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box low-risk">
                ✅ Low Risk of Heart Disease
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Confidence Score")
        
        if prediction == 1:
            confidence = probability[1] * 100
            st.metric(
                "Risk Probability",
                f"{confidence:.1f}%",
                delta="High Risk"
            )
        else:
            confidence = probability[0] * 100
            st.metric(
                "Safety Probability",
                f"{confidence:.1f}%",
                delta="Low Risk"
            )
    
    # Display probability breakdown
    st.subheader("📈 Probability Breakdown")
    
    prob_df = pd.DataFrame({
        'Category': ['No Heart Disease', 'Heart Disease'],
        'Probability': probability
    })
    
    # Create a progress bar visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("No Heart Disease Probability:")
        st.progress(probability[0])
        st.write(f"{probability[0]*100:.1f}%")
    
    with col2:
        st.write("Heart Disease Probability:")
        st.progress(probability[1])
        st.write(f"{probability[1]*100:.1f}%")
    
    # Add disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Disclaimer**: This prediction is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.
    """)

def display_feature_info():
    """Display information about the features used"""
    with st.expander("ℹ️ Feature Information"):
        st.markdown("""
        ### Feature Descriptions
        
        1. **Age**: Patient age in years
        2. **Sex**: Gender (1=Male, 0=Female)
        3. **Chest Pain Type**: Type of chest pain (0-3, representing different types)
        4. **Resting Blood Pressure**: Blood pressure in mm Hg
        5. **Serum Cholesterol**: Cholesterol level in mg/dl
        6. **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl
        7. **Resting ECG**: Resting electrocardiogram results
        8. **Max Heart Rate**: Maximum heart rate achieved during exercise
        9. **Exercise Angina**: Whether exercise induced angina
        10. **ST Depression**: ST depression induced by exercise
        11. **ST Slope**: Slope of peak exercise ST segment
        12. **Major Vessels**: Number of major vessels colored by fluoroscopy
        13. **Thalassemia**: Type of thalassemia
        """)

def main():
    """Main function to run the Streamlit app"""
    
    # Display header
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction App</h1>
        <p>Machine Learning powered prediction system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'disease.pkl' exists.")
        return
    
    # Get user input
    features = get_user_input()
    
    # Display user input summary
    st.subheader("📋 Input Summary")
    
    # Create a dataframe for display
    feature_names = [
        'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
        'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
        'Exercise Angina', 'ST Depression', 'ST Slope', 
        'Major Vessels', 'Thalassemia'
    ]
    
    input_data = {
        'Feature': feature_names,
        'Value': features
    }
    
    # Convert values to more readable format for display
    readable_values = []
    for i, (name, value) in enumerate(zip(feature_names, features)):
        if name == 'Sex':
            readable_values.append('Male' if value == 1 else 'Female')
        elif name == 'Fasting Blood Sugar':
            readable_values.append('Yes' if value == 1 else 'No')
        elif name == 'Exercise Angina':
            readable_values.append('Yes' if value == 1 else 'No')
        elif name == 'Chest Pain Type':
            cp_types = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            readable_values.append(cp_types[int(value)])
        elif name == 'Resting ECG':
            ecg_types = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            readable_values.append(ecg_types[int(value)])
        elif name == 'ST Slope':
            slope_types = ["Upsloping", "Flat", "Downsloping"]
            readable_values.append(slope_types[int(value)])
        elif name == 'Thalassemia':
            thal_types = ["Normal", "Fixed Defect", "Reversable Defect"]
            readable_values.append(thal_types[int(value)])
        else:
            readable_values.append(value)
    
    summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': readable_values
    })
    
    st.dataframe(summary_df, hide_index=True)
    
    # Make prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            prediction, probability = make_prediction(model, features)
            
            if prediction is not None:
                # Display results
                display_results(prediction, probability)
    
    # Display feature information
    display_feature_info()
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Built with ❤️ using Streamlit and Scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

