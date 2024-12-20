import streamlit as st
import pandas as pd
import joblib
import warnings

# Load model
model_file = "/Users/d/Desktop/Breast_Cancer_Project/knn_model.joblib" 
with open(model_file, 'rb') as f:
    knn = joblib.load(f)

scaler_file = "scaler.joblib"  
scaler = joblib.load(scaler_file)
red_features = [
    'diagnosis', 'concave points_worst', 'perimeter_worst',
    'concave points_mean', 'radius_worst', 'perimeter_mean',
    'area_worst', 'radius_mean', 'area_mean', 'concavity_mean',
    'concavity_worst', 'compactness_mean', 'compactness_worst',
    'radius_se', 'perimeter_se', 'area_se'
]


default_values = {
    'radius_mean': 0.000,
    'texture_mean': 0.000,
    'perimeter_mean': 0.000,
    'area_mean': 0.000,
    'smoothness_mean': 0.000,
    'compactness_mean': 0.000,
    'concavity_mean': 0.000,
    'concave points_mean': 0.000,
    'symmetry_mean': 0.000,
    'fractal_dimension_mean': 0.000,
    'radius_se': 0.000,
    'texture_se': 0.000,
    'perimeter_se': 0.000,
    'area_se': 0.000,
    'smoothness_se': 0.000,
    'compactness_se': 0.000,
    'concavity_se': 0.000,
    'concave points_se': 0.000,
    'symmetry_se': 0.000,
    'fractal_dimension_se': 0.000,
    'radius_worst': 0.000,
    'texture_worst': 0.000,
    'perimeter_worst': 0.000,
    'area_worst': 0.000,
    'smoothness_worst': 0.000,
    'compactness_worst': 0.000,
    'concavity_worst': 0.000,
    'concave points_worst': 0.000,
    'symmetry_worst': 0.000,
    'fractal_dimension_worst': 0.000
}

# Sidebar for patient inputs
st.sidebar.image("/Users/d/Desktop/Breast_Cancer_Project/Detection_Breast_Cancer.jpg", use_container_width=True)
st.sidebar.header("Patient Inputs")

# Collect patient inputs with red color for specific features
patient_inputs = {}
for feature, value in default_values.items():
    if feature in red_features:
        # Highlight important features with a warning (styled in red)
        st.sidebar.warning(f"**{feature}** (Important!)")  # Red style in warning
    patient_inputs[feature] = st.sidebar.number_input(label=feature, value=value, step=0.001)

input_df = pd.DataFrame([patient_inputs])

# Display images and title
st.image("//Users/d/Desktop/Breast_Cancer_Project/AI_diagnosis.jpeg", use_container_width=True)
st.title("Breast Cancer Detection")

# Layout with columns
left_col, right_col = st.columns(2)  

# Use right_col for the diagnosis button and result
with right_col:
    st.subheader("Diagnosis")

    # Perform the diagnosis when the button is clicked
    if st.button("Diagnosis"):
        scaled_input = scaler.transform(input_df)

# Predict diagnosis (1: Malignant, 0: Benign)
        prediction = knn.predict(scaled_input)
        prediction_proba = knn.predict_proba(scaled_input)

        predicted_probability = prediction_proba[0, prediction[0]]
        percentage = predicted_probability * 100
        if prediction[0] == 1:
            st.warning(f"Malignant (Probability: {percentage:.2f}%)")
        else:
            st.success(f"Benign (Probability: {percentage:.2f}%)")
# Suppress the warning
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

#   /Users/d/Desktop/Breast_Cancer_Project/project.py