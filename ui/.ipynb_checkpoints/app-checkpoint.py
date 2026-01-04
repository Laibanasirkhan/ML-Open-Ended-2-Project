import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.preprocessing import MinMaxScaler

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
    page_icon="❤️",
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
.css-1d391kg {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    border-right: 3px solid #ffffff;
    border-radius: 10px;
    padding: 20px;
}
h1, h2, h3 { color: #ffffff; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-weight:bold; }
.stButton>button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    border-radius: 25px;
    border: none;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
}
.stSlider, .stSelectbox { background-color: rgba(255,255,255,0.1); border-radius: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.3); }
hr { border: 2px solid rgba(255,255,255,0.5); border-radius: 5px; }
.stAlert { border-radius: 15px; border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.plotly-graph-div { border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.3); background-color: rgba(255,255,255,0.9); }
.dataframe { border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.3); background-color: rgba(255,255,255,0.9); color:black; }
.footer { position: fixed; bottom: 0; width: 100%; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); text-align: center; padding: 15px; font-size: 14px; color: #ffffff; border-top: 2px solid rgba(255,255,255,0.3); }
.card { background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); backdrop-filter: blur(10px); }
.metric { background: rgba(255,255,255,0.2); border-radius: 10px; padding: 15px; text-align: center; margin: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
@media (max-width: 768px) {
    .stApp { font-size: 14px; }
    .card { padding: 15px; }
}
.fade-in { animation: fadeIn 1s ease-in; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

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

# ------------------ UTILITY FUNCTIONS ------------------
def load_lottieurl(url: str):
    if not LOTTIE_AVAILABLE:
        return None
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def section_header(title, subtitle=""):
    st.markdown(f"<div class='card fade-in'><h2>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: #ffffff; font-size: 18px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("<h3 style='color: #ffffff;'>🩺 Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "Select Page",
    ["Home 🏠", "Prediction 🩺", "Data Analysis 📊", "About ℹ"],
    key="nav"
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home 🏠":
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""
        <div class='card'>
        <h1>❤️ Heart Disease Risk Assessment</h1>
        <h3>Predict. Analyze. Understand.</h3>
        <p>Heart disease is a <strong>global health challenge</strong> that often develops silently. Identifying risk factors early can help prevent severe outcomes and support better clinical decision-making.</p>
        <p>This application applies <strong>machine learning on real clinical data</strong> to estimate an individual’s <strong>risk of heart disease</strong> using key health indicators.</p>
        <hr>
        <h4>🔍 What You Can Do Here</h4>
        <ul>
            <li><strong>Predict</strong> heart disease risk in real time</li>
            <li><strong>Analyze</strong> patient health patterns visually</li>
            <li><strong>Compare</strong> patient data with population averages</li>
            <li><strong>Explore</strong> applied machine learning in healthcare</li>
        </ul>
        <p>Built as a <strong>high-quality data science portfolio project</strong>, demonstrating practical ML deployment and healthcare analytics.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>📊 Quick Dataset Stats</h3></div>", unsafe_allow_html=True)
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
elif page == "Prediction 🩺":
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

        submitted = st.form_submit_button("🚀 Predict Risk")

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
            st.error(f"⚠ **High Risk of Heart Disease** ({probability:.2f}%)")
            st.markdown("<div class='card'><p><strong>Recommendation:</strong> Consult a healthcare professional immediately.</p></div>", unsafe_allow_html=True)
        else:
            st.success(f"✅ **Low Risk of Heart Disease** ({probability:.2f}%)")
            st.markdown("<div class='card'><p><strong>Note:</strong> Maintain a healthy lifestyle.</p></div>", unsafe_allow_html=True)

# ==================================================
# DATA ANALYSIS PAGE
# ==================================================
elif page == "Data Analysis 📊":
    section_header("Patient Profile Analysis", "Compare your inputs with dataset averages.")
    if "patient_data" not in st.session_state:
        st.info("💡 Please make a prediction first to view analysis.")
        st.stop()

    patient = st.session_state["patient_data"]
    dataset_avg = df[patient.columns].mean()

    # Radar Chart
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
        title="Normalized Patient Profile vs Dataset Average",
        color_discrete_map={"Patient":"#ff3b3b","Dataset Average":"#00ffcc"}
    )
    fig.update_traces(fill='toself', line=dict(width=4))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=patient.columns, y=patient.iloc[0].values, name='Patient', marker_color='#ff6b6b',
                             text=[f"{v:.1f}" if isinstance(v,(int,float)) else str(v) for v in patient.iloc[0].values], textposition='auto'))
    bar_fig.add_trace(go.Bar(x=patient.columns, y=dataset_avg.values, name='Dataset Avg', marker_color='#4ecdc4',
                             text=[f"{v:.1f}" for v in dataset_avg.values], textposition='auto'))
    bar_fig.update_layout(title="Feature Comparison: Patient vs Dataset Average",
                          xaxis_title="Clinical Features",
                          yaxis_title="Value",
                          barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(bar_fig, use_container_width=True)

    # Pie Chart
    risk_counts = df['target'].value_counts()
    pie_fig = px.pie(
        values=risk_counts.values,
        names=['Low Risk','High Risk'],
        title="Dataset Risk Distribution",
        color_discrete_sequence=['#4ecdc4','#ff6b6b']
    )
    pie_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(pie_fig, use_container_width=True)

    # Patient Summary Table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Patient Input Summary</h3>", unsafe_allow_html=True)

    patient_summary = patient.T.rename(columns={0: "Patient Value"})

    # Format numeric and categorical values
    def format_value(x):
        try:
            return f"{x:.1f}"
        except:
            return str(x)
    patient_summary["Patient Value"] = patient_summary["Patient Value"].apply(format_value)

    # Display table with clean white background & black text
    st.table(
        patient_summary.style.set_table_styles([
            {'selector':'thead', 'props':[('background-color','#4ecdc4'),
                                          ('color','white'),
                                          ('font-weight','bold'),
                                          ('text-align','center')]},
            {'selector':'tbody', 'props':[('background-color','white'),
                                          ('color','black'),
                                          ('text-align','center'),
                                          ('font-size','16px')]}
        ])
    )

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
