import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF

# Optional: Lottie animations
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* ---------- SIDEBAR (INTEGRATED LOOK) ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%) !important;
    border-right: 3px solid rgba(255,255,255,0.4);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] select {
    background-color: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border-radius: 10px;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    border-radius: 25px;
    border: none;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    transform: translateY(-2px);
}

/* ---------- CARDS ---------- */
.card {
    background: rgba(255,255,255,0.12);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

/* ---------- METRICS ---------- */
.metric {
    background: rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

/* ---------- TABLE ---------- */
.table-container {
    background-color: rgba(0,0,0,0.85);
    border-radius: 15px;
    padding: 20px;
    color: #ffff00;
    font-weight: bold;
}

/* ---------- ALERT ---------- */
.dark-alert {
    background-color: #8b0000;
    color: white;
    padding: 15px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: bold;
}

/* ---------- HEART STICKER ---------- */
.heart-sticker {
    background: white;
    border-radius: 50%;
    padding: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
}
.heart-sticker img {
    width: 260px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_cleaned.csv")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "real_human_heart.png")

# ------------------ LOAD DATA & MODEL ------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

df = load_data()
model, scaler = load_model()

# ------------------ UTILITIES ------------------
def section_header(title, subtitle=""):
    st.markdown(f"<div class='card'><h2>{title}</h2><p>{subtitle}</p></div>", unsafe_allow_html=True)

def generate_pdf(patient_data, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Heart Disease Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {probability:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Patient Data:", ln=True)
    for col in patient_data.columns:
        pdf.cell(200, 10, txt=f"{col}: {patient_data[col].iloc[0]}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## 🩺 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Prediction", "Data Analysis", "About ℹ"]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""
        <div class='card'>
        <h1>🫀 Heart Disease Risk Assessment</h1>
        <h3>Predict. Analyze. Understand.</h3>
        <p>Heart disease is a <strong>global health challenge</strong> that often develops silently. Identifying risk factors early can help prevent severe outcomes and support better clinical decision-making.</p>
        <p>This application applies <strong>machine learning on real clinical data</strong> to estimate an individual’s <strong>risk of heart disease</strong> using key health indicators.</p>
        <hr>
        <h4> What You Can Do Here</h4>
        <ul>
            <li><strong>Predict</strong> heart disease risk in real time</li>
            <li><strong>Analyze</strong> patient health patterns visually</li>
            <li><strong>Compare</strong> patient data with population averages</li>
            <li><strong>Explore</strong> applied machine learning in healthcare</li>
        </ul>
        <p>Built as a <strong>high-quality data science portfolio project</strong>, demonstrating practical ML deployment and healthcare analytics.</p>
        </div>
        """, unsafe_allow_html=True)
        col1_stat, col2_stat, col3_stat, col4_stat = st.columns(4)
        with col1_stat:
            st.markdown(f"<div class='metric'><h4>{len(df)}</h4><p>Total Patients</p></div>", unsafe_allow_html=True)
        with col2_stat:
            st.markdown(f"<div class='metric'><h4>{df.age.mean():.1f}</h4><p>Avg Age (years)</p></div>", unsafe_allow_html=True)
        with col3_stat:
            st.markdown(f"<div class='metric'><h4>{df[df.target == 1].shape[0]}</h4><p>High Risk Cases</p></div>", unsafe_allow_html=True)
        with col4_stat:
            st.markdown(f"<div class='metric'><h4>{df.cholesterol.mean():.0f}</h4><p>Avg Cholesterol (mg/dl)</p></div>", unsafe_allow_html=True)

    with col2:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH, use_container_width=True, caption="Heart Health Matters")
        st.markdown("""
            <div class='card' style='font-size:18px; font-style:italic; text-align:center;'>
                🩺 <strong>"Take care of your heart, and it will take care of you."</strong>
            </div>
        """, unsafe_allow_html=True)
# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Prediction":
    section_header("Heart Disease Prediction", "Enter patient details below to assess risk.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age (years)", int(df.age.min()), int(df.age.max()), 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            chest_pain_type = st.selectbox("Chest Pain Type", sorted(df.chest_pain_type.unique()))
            resting_bp_s = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
            cholesterol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
            fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            resting_ecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x]
            )
            max_heart_rate = st.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", sorted(df.st_slope.unique()))

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_bp_s, cholesterol,
                                    fasting_blood_sugar, resting_ecg, max_heart_rate,
                                    exercise_angina, oldpeak, st_slope]],
                                  columns=['age','sex','chest_pain_type','resting_bp_s','cholesterol',
                                           'fasting_blood_sugar','resting_ecg','max_heart_rate',
                                           'exercise_angina','oldpeak','st_slope'])

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i+1)
        progress_bar.empty()

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]*100
        st.session_state["patient_data"] = input_data

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Heart Disease Risk (%)"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'color': "#ff6b6b" if prediction==1 else "#4ecdc4"},
                   'steps':[{'range':[0,30],'color':'#4ecdc4'},
                            {'range':[30,70],'color':'#ffa500'},
                            {'range':[70,100],'color':'#ff6b6b'}]}
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prediction==1:
            st.markdown(f"<div class='dark-alert'>⚠ High Risk of Heart Disease ({probability:.2f}%)</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><p><strong>Recommendation:</strong> Consult a healthcare professional immediately.</p></div>", unsafe_allow_html=True)
        else:
            st.success(f"✅ Low Risk of Heart Disease ({probability:.2f}%)")
            st.markdown("<div class='card'><p><strong>Note:</strong> Maintain a healthy lifestyle.</p></div>", unsafe_allow_html=True)

        # PDF Download
        pdf_data = generate_pdf(input_data, prediction, probability)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_data,
            file_name="heart_disease_report.pdf",
            mime="application/pdf"
        )

# ==================================================
# DATA ANALYSIS PAGE
# ==================================================
elif page == "Data Analysis":
    section_header("Patient Profile Analysis", "Compare your inputs with dataset averages.")
    if "patient_data" not in st.session_state:
        st.info("Please make a prediction first to view analysis.")
        st.stop()

    patient = st.session_state["patient_data"]
    dataset_avg = df[patient.columns].mean()

    # Enhanced Radar Chart with bright colors and no white background
    scaler_radar = MinMaxScaler()
    radar_data = pd.DataFrame({
        "Feature": patient.columns,
        "Patient": scaler_radar.fit_transform(patient.T).flatten(),
        "Dataset Average": scaler_radar.fit_transform(dataset_avg.values.reshape(-1,1)).flatten()
    })
    fig = px.line_polar(
        radar_data.melt(id_vars="Feature"),
        r="value",
        theta="Feature",
        color="variable",
        line_close=True,
        title="<b>Normalized Patient Profile vs Dataset Average</b>",
        color_discrete_map={"Patient":"#ff0000","Dataset Average":"#00ff00"}
    )
    fig.update_traces(fill='toself', line=dict(width=4))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    # Enhanced Bar Chart with bright colors and visible text (white on dark)
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=patient.columns, y=patient.iloc[0].values, name='Patient', marker_color='#ff0000',
                             text=[f"{v:.1f}" if isinstance(v,(int,float)) else str(v) for v in patient.iloc[0].values], textposition='auto', textfont=dict(color='white')))
    bar_fig.add_trace(go.Bar(x=patient.columns, y=dataset_avg.values, name='Dataset Avg', marker_color='#00ff00',
                             text=[f"{v:.1f}" for v in dataset_avg.values], textposition='auto', textfont=dict(color='white')))
    bar_fig.update_layout(title="<b>Feature Comparison: Patient vs Dataset Average</b>",
                          xaxis_title="Clinical Features",
                          yaxis_title="Value",
                          barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(bar_fig, use_container_width=True)

    # Enhanced Pie Chart with bright colors
    risk_counts = df['target'].value_counts()
    pie_fig = px.pie(
        values=risk_counts.values,
        names=['Low Risk','High Risk'],
        title="<b>Dataset Risk Distribution</b>",
        color_discrete_sequence=['#00ff00','#ff0000']
    )
    pie_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(pie_fig, use_container_width=True)

    # Enhanced Patient Summary Table with better styling
        # Enhanced Patient Summary Table with better styling
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Patient Input Summary</h3>", unsafe_allow_html=True)

    patient_summary = patient.T.rename(columns={0: "Patient Value"})

    # Format numeric and categorical values
    def format_value(x):
        if isinstance(x, float):
            return f"{x:.1f}"
        elif isinstance(x, int):
            return str(x)
        else:
            return str(x)
    patient_summary["Patient Value"] = patient_summary["Patient Value"].apply(format_value)

    # Display table with enhanced styling for visibility
    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
    st.table(patient_summary)
    st.markdown("</div>", unsafe_allow_html=True)
# ==================================================
# DATA ANALYSIS PAGE
# ==================================================
elif page == "Data Analysis":
    section_header("Patient Profile Analysis", "Compare your inputs with dataset averages.")
    if "patient_data" not in st.session_state:
        st.info("Please make a prediction first to view analysis.")
        st.stop()

    patient = st.session_state["patient_data"]
    dataset_avg = df[patient.columns].mean()

    # Enhanced Radar Chart with bright colors and no white background
    scaler_radar = MinMaxScaler()
    radar_data = pd.DataFrame({
        "Feature": patient.columns,
        "Patient": scaler_radar.fit_transform(patient.T).flatten(),
        "Dataset Average": scaler_radar.fit_transform(dataset_avg.values.reshape(-1,1)).flatten()
    })
    fig = px.line_polar(
        radar_data.melt(id_vars="Feature"),
        r="value",
        theta="Feature",
        color="variable",
        line_close=True,
        title="<b>Normalized Patient Profile vs Dataset Average</b>",
        color_discrete_map={"Patient":"#ff0000","Dataset Average":"#00ff00"}
    )
    fig.update_traces(fill='toself', line=dict(width=4))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    # Enhanced Bar Chart with bright colors and visible text (white on dark)
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=patient.columns, y=patient.iloc[0].values, name='Patient', marker_color='#ff0000',
                             text=[f"{v:.1f}" if isinstance(v,(int,float)) else str(v) for v in patient.iloc[0].values], textposition='auto', textfont=dict(color='white')))
    bar_fig.add_trace(go.Bar(x=patient.columns, y=dataset_avg.values, name='Dataset Avg', marker_color='#00ff00',
                             text=[f"{v:.1f}" for v in dataset_avg.values], textposition='auto', textfont=dict(color='white')))
    bar_fig.update_layout(title="<b>Feature Comparison: Patient vs Dataset Average</b>",
                          xaxis_title="Clinical Features",
                          yaxis_title="Value",
                          barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(bar_fig, use_container_width=True)

    # Enhanced Pie Chart with bright colors
    risk_counts = df['target'].value_counts()
    pie_fig = px.pie(
        values=risk_counts.values,
        names=['Low Risk','High Risk'],
        title="<b>Dataset Risk Distribution</b>",
        color_discrete_sequence=['#00ff00','#ff0000']
    )
    pie_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(pie_fig, use_container_width=True)

    # Enhanced Patient Summary Table with better styling
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Patient Input Summary</h3>", unsafe_allow_html=True)

    patient_summary = patient.T.rename(columns={0: "Patient Value"})

    # Format numeric and categorical values
    def format_value(x):
        if isinstance(x, float):
            return f"{x:.1f}"
        elif isinstance(x, int):
            return str(x)
        else:
            return str(x)
    patient_summary["Patient Value"] = patient_summary["Patient Value"].apply(format_value)

    # Display table with enhanced styling for visibility
    st.markdown("<div class='table-container'>", unsafe_allow_html=True)
    st.table(patient_summary)
    st.markdown("</div>", unsafe_allow_html=True)
# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "About ℹ":
    section_header("About This Project")
    st.markdown("""
    <div class='card'>
    <h3> Purpose</h3>
    <p>This application predicts heart disease risk using machine learning models trained on real clinical datasets. It helps identify risk factors early.</p>
    <h3> Key Features</h3>
    <ul>
        <li>Real-time heart disease risk prediction</li>
        <li>Interactive data analysis and visualization</li>
        <li>Comparison with population averages</li>
        <li>Modern, responsive, visually appealing UI</li>
    </ul>
    <h3> Technology Stack</h3>
    <p>Python, Streamlit, Plotly, Scikit-learn, Pandas, Lottie Animations (optional)</p>
    <h3> Why It’s Impressive</h3>
    <p>Demonstrates practical deployment of ML in healthcare, clean UI/UX design, and advanced visualization for portfolio showcase.</p>
    </div>
    """, unsafe_allow_html=True)
