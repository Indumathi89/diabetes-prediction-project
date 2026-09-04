import streamlit as st
import pandas as pd
import joblib

# Load the trained model
data = joblib.load("diabetes_model.pkl")

model = data["model"]
fill_values = data["fill_values"]

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details below to predict the diabetes outcome.")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# Prediction button
st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Apply the same preprocessing used during training
    columns_to_fix = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in columns_to_fix:
        if input_data[column].iloc[0] == 0:
            input_data[column] = fill_values[column]

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The model predicts: Diabetes")
    else:
        st.success("✅ The model predicts: No Diabetes")

    st.info("Model used: Random Forest Classifier | Test Accuracy: 75.32%")

st.divider()

st.caption(
    "This application is developed for educational purposes only "
    "and should not be used as a medical diagnosis."
)