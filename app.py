import streamlit as st
import pandas as pd
import pickle


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

with open("disease_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🩺 Disease Prediction System")

st.write(
    "Enter the patient's health information "
    "to predict whether disease is present."
)

st.divider()


# ==========================================
# PATIENT INFORMATION
# ==========================================

st.subheader("Patient Information")

col1, col2 = st.columns(2)


# ---------- LEFT COLUMN ----------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=80,
        max_value=200,
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=350,
        value=200,
        step=1
    )


# ---------- RIGHT COLUMN ----------

with col2:

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=50,
        max_value=250,
        value=100,
        step=1
    )

    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=40,
        max_value=200,
        value=70,
        step=1
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    exercise = st.selectbox(
        "Exercise Level",
        ["Low", "Medium", "High"]
    )


st.divider()


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🔍 Predict Disease",
    use_container_width=True
):

    # --------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "BMI": [bmi],
        "Blood_Pressure_mmHg": [blood_pressure],
        "Cholesterol_mg_dL": [cholesterol],
        "Glucose_mg_dL": [glucose],
        "Heart_Rate_bpm": [heart_rate],
        "Smoking": [smoking],
        "Exercise_Level": [exercise]
    })


    # --------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------

    input_data = pd.get_dummies(input_data)


    # --------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------
    # MAKE PREDICTION
    # --------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------
    # DISPLAY RESULT
    # --------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Prediction: Disease"
        )

    else:

        st.success(
            "✅ Prediction: No Disease"
        )


    st.info(
        "This application is an educational machine-learning "
        "project and should not be considered a medical diagnosis."
    )