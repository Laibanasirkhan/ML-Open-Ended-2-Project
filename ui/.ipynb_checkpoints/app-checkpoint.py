import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="wide",
    page_icon="❤️"
)

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_cleaned.csv")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "heart.png")

# ------------------ LOAD DATA & MODEL ------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

df = load_data()
model, scaler = load_model()

# ------------------ HEADER ------------------
def section_header(title):
    st.markdown(f"## {title}")
    st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Home 🏠", "Prediction 🩺", "Data Analysis 📊", "About ℹ"]
)

# HOME
if page == "Home 🏠":

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        # ❤️ Heart Disease Risk Assessment

        ### Predict. Analyze. Understand.

        Heart disease is a **global health challenge** that often develops silently.
        Identifying risk factors early can help prevent severe outcomes and support
        better clinical decision-making.

        This application applies **machine learning on real clinical data**
        to estimate an individual’s **risk of heart disease** using key
        health indicators.

        ---
        #### 🔍 What You Can Do Here
        - **Predict** heart disease risk in real time  
        - **Analyze** patient health patterns visually  
        - **Compare** patient data with population averages  
        - **Explore** applied machine learning in healthcare  

        Built as a **high-quality data science portfolio project**,
        demonstrating practical ML deployment and healthcare analytics.
        """)

    with col2:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH, use_container_width=True)

# ==================================================
# PREDICTION
# ==================================================
elif page == "Prediction 🩺":

    section_header("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", int(df.age.min()), int(df.age.max()), 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chest_pain_type = st.selectbox("Chest Pain Type", sorted(df.chest_pain_type.unique()))
        resting_bp_s = st.slider("Resting Blood Pressure", 80, 200, 120)
        cholesterol = st.slider("Cholesterol", 100, 400, 200)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )
        max_heart_rate = st.slider("Maximum Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", sorted(df.st_slope.unique()))

    input_data = pd.DataFrame([[
        age, sex, chest_pain_type, resting_bp_s, cholesterol,
        fasting_blood_sugar, resting_ecg, max_heart_rate,
        exercise_angina, oldpeak, st_slope
    ]], columns=[
        'age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_angina', 'oldpeak', 'st_slope'
    ])

    if st.button("Predict Risk"):
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1] * 100

        st.session_state["patient_data"] = input_data

        if prediction == 1:
            st.error(f"⚠ High Risk of Heart Disease ({probability:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({probability:.2f}%)")

# ==================================================
# DATA ANALYSIS
# ==================================================
elif page == "Data Analysis 📊":

    section_header("Patient Profile Analysis")

    if "patient_data" not in st.session_state:
        st.info("Please make a prediction first to view analysis.")
        st.stop()

    patient = st.session_state["patient_data"]
    dataset_avg = df[patient.columns].mean()

    radar_df = pd.DataFrame({
        "Feature": patient.columns,
        "Patient": patient.iloc[0].values,
        "Dataset Average": dataset_avg.values
    })

    fig = px.line_polar(
        radar_df.melt(id_vars="Feature"),
        r="value",
        theta="Feature",
        color="variable",
        line_close=True,
        title="Patient Profile vs Dataset Average"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(patient, use_container_width=True)

# ==================================================
# ABOUT
# ==================================================
elif page == "About ℹ":

    section_header("About This Project")

    st.markdown("""
    ### Heart Disease Risk Assessment System

    This project is an end-to-end **machine learning application** built using
    a cleaned and standardized heart disease dataset.

    The system allows users to input patient health indicators and receive
    a **probability-based risk prediction** for heart disease.

    #### Technologies Used
    - Python, Pandas, NumPy  
    - Scikit-learn  
    - Plotly  
    - Streamlit  

    ---
    ### ⚠ Medical Disclaimer

    This application is intended **solely for educational, academic, and demonstration purposes**.

    The predictions generated by this system **do not constitute medical advice, diagnosis,
    or treatment**.

    No clinical decisions should be made based on this application.
    Always consult a **qualified healthcare professional** for medical concerns.
    """)



