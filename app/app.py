"""
app.py — Streamlit Web App for Student Placement Prediction
Run: streamlit run app/app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
}
.hero-banner h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero-banner p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1.05rem; }

.section-title {
    color: #a78bfa;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(167,139,250,0.3);
}

.result-placed {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(56, 239, 125, 0.35);
}
.result-not-placed {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(231, 76, 60, 0.35);
}
.result-label      { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
.result-confidence { font-size: 1.1rem; opacity: 0.9; }
.result-emoji      { font-size: 3.5rem; margin-bottom: 0.8rem; }

.rec-item {
    background: rgba(167, 139, 250, 0.1);
    border-left: 3px solid #a78bfa;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    color: #e2e8f0;
    font-size: 0.92rem;
}
.metric-box   { background: rgba(255,255,255,0.07); border-radius: 10px; padding: 1rem;
                text-align: center; border: 1px solid rgba(255,255,255,0.1); }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }

.dataset-note {
    background: rgba(234,179,8,0.1);
    border: 1px solid rgba(234,179,8,0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #fde68a;
    font-size: 0.85rem;
    margin-top: 1rem;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts ───────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(MODEL_DIR, 'mlp_model.pkl'))
    scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    branch_cols   = joblib.load(os.path.join(MODEL_DIR, 'branch_cols.pkl'))
    return model, scaler, feature_names, branch_cols

try:
    model, scaler, feature_names, branch_cols = load_artifacts()
    model_loaded = True
    # Derive branch options from saved column names (branch_CSE → CSE)
    branch_options = sorted([c.replace('branch_', '') for c in branch_cols])
except Exception as e:
    model_loaded = False
    load_error   = str(e)
    branch_options = ['CSE', 'IT', 'ECE', 'EEE', 'Mechanical', 'Civil']


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 About")
    st.markdown("""
This app uses a **Multi-Layer Perceptron (MLP)** trained on 100,000 student records
to predict campus placement probability.

**Model:** scikit-learn MLPClassifier  
**Dataset:** Student Placement Prediction Dataset 2026 (Kaggle)  
**Preprocessing:** OHE for branch, ordinal for tier, split-then-scale
""")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model_loaded:
        st.success("✅ Model loaded")
        st.caption(f"Architecture: {model.hidden_layer_sizes}")
        st.caption(f"Features: {len(feature_names)}")
    else:
        st.error("❌ Model not found — run the notebook first")
    st.markdown("""
<div class="dataset-note">
⚠️ <strong>Note:</strong> This dataset has synthetic labels with max feature correlation of 0.082. 
Model accuracy ceiling is ~57% for any algorithm — this is a dataset characteristic, not a model flaw.
</div>
""", unsafe_allow_html=True)


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🎓 Student Placement Predictor</h1>
    <p>Enter your academic & professional profile to predict your campus placement outcome</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model: {load_error}")
    st.info("Run `notebooks/model_training.ipynb` first to train and save the model.")
    st.stop()


# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Your Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-title">🎓 Academic Profile</div>', unsafe_allow_html=True)
    age        = st.slider("Age", 18, 30, 21, key="age")
    cgpa       = st.slider("CGPA", 4.5, 10.0, 7.5, step=0.1, key="cgpa")
    attendance = st.slider("Attendance (%)", 50, 100, 80, key="attend")
    backlogs   = st.number_input("Number of Backlogs", 0, 10, 0, key="backlogs")
    study_hrs  = st.slider("Study Hours / Day", 0, 12, 4, key="study")
    sleep_hrs  = st.slider("Sleep Hours / Day", 4, 10, 7, key="sleep")

with col2:
    st.markdown('<div class="section-title">💻 Technical Skills</div>', unsafe_allow_html=True)
    coding_score   = st.slider("Coding Skill Score (0–100)", 0, 100, 65, key="coding")
    aptitude_score = st.slider("Aptitude Score (0–100)", 0, 100, 65, key="apt")
    logical_score  = st.slider("Logical Reasoning Score (0–100)", 0, 100, 65, key="logic")
    certifications = st.number_input("Certifications Count", 0, 20, 2, key="certs")
    projects_count = st.number_input("Projects Count", 0, 20, 3, key="proj")
    github_repos   = st.number_input("GitHub Repositories", 0, 50, 5, key="github")
    internships    = st.number_input("Internships Count", 0, 10, 1, key="intern")
    hackathons     = st.number_input("Hackathons Participated", 0, 20, 1, key="hack")

with col3:
    st.markdown('<div class="section-title">🗣️ Professional Profile</div>', unsafe_allow_html=True)
    comm_score    = st.slider("Communication Skill (0–100)", 0, 100, 65, key="comm")
    mock_score    = st.slider("Mock Interview Score (0–100)", 0, 100, 65, key="mock")
    leadership    = st.slider("Leadership Score (0–100)", 0, 100, 60, key="lead")
    extracurr     = st.slider("Extracurricular Score (0–100)", 0, 100, 55, key="extra")
    linkedin_conn = st.number_input("LinkedIn Connections", 0, 3000, 200, key="linkedin")
    gender        = st.selectbox("Gender", ["Male", "Female"], key="gender")
    branch        = st.selectbox("Branch", branch_options, key="branch")
    college_tier  = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"], key="ctier")
    volunteer     = st.selectbox("Volunteer Experience", ["Yes", "No"], key="vol")


# ── Build Feature Vector (must exactly match notebook preprocessing) ───────────
def build_feature_vector() -> pd.DataFrame:
    """
    Replicates the exact preprocessing pipeline from model_training.ipynb:
      - gender         → Male=1, Female=0
      - volunteer_exp  → Yes=1, No=0
      - college_tier   → Tier 1=2, Tier 2=1, Tier 3=0 (ordinal)
      - branch         → one-hot, column names: branch_<name>
    Returns a DataFrame with columns matching feature_names.
    """
    tier_map = {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}

    # Start with all feature names, defaulting to 0
    row = {f: 0 for f in feature_names}

    # Numeric fields
    row['age']                        = age
    row['cgpa']                       = cgpa
    row['attendance_percentage']      = attendance
    row['backlogs']                   = backlogs
    row['study_hours_per_day']        = study_hrs
    row['sleep_hours']                = sleep_hrs
    row['coding_skill_score']         = coding_score
    row['aptitude_score']             = aptitude_score
    row['logical_reasoning_score']    = logical_score
    row['certifications_count']       = certifications
    row['projects_count']             = projects_count
    row['github_repos']               = github_repos
    row['internships_count']          = internships
    row['hackathons_participated']    = hackathons
    row['communication_skill_score']  = comm_score
    row['mock_interview_score']       = mock_score
    row['linkedin_connections']       = linkedin_conn
    row['extracurricular_score']      = extracurr
    row['leadership_score']           = leadership

    # Encoded categoricals
    row['gender']               = 1 if gender == 'Male' else 0
    row['volunteer_experience'] = 1 if volunteer == 'Yes' else 0
    row['college_tier']         = tier_map[college_tier]

    # One-hot: set the selected branch column to 1
    branch_col = f'branch_{branch}'
    if branch_col in row:
        row[branch_col] = 1

    # Return as DataFrame so sklearn doesn't warn about feature names
    return pd.DataFrame([row])[feature_names]


# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _, _ = st.columns([1, 2, 2])
with predict_col:
    predict_btn = st.button("🔮 Predict Placement", type="primary", use_container_width=True)

if predict_btn:
    feat_df     = build_feature_vector()
    feat_scaled = scaler.transform(feat_df)  # No warning — DataFrame has named columns
    prediction  = model.predict(feat_scaled)[0]
    proba       = model.predict_proba(feat_scaled)[0]

    # proba[0]=Not Placed, proba[1]=Placed
    placed_prob    = proba[1] * 100
    not_placed_prob = proba[0] * 100
    is_placed       = (prediction == 1)

    st.markdown("---")
    st.markdown("### 🏆 Prediction Result")

    res_col, rec_col = st.columns([1, 1])

    with res_col:
        if is_placed:
            st.markdown(f"""
<div class="result-placed">
    <div class="result-emoji">🎉</div>
    <div class="result-label">LIKELY PLACED</div>
    <div class="result-confidence">Placement Probability: {placed_prob:.1f}%</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-not-placed">
    <div class="result-emoji">📚</div>
    <div class="result-label">MORE WORK NEEDED</div>
    <div class="result-confidence">Placement Probability: {placed_prob:.1f}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{cgpa}</div><div class="metric-label">CGPA</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{coding_score}</div><div class="metric-label">Coding</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{mock_score}</div><div class="metric-label">Mock Int.</div></div>', unsafe_allow_html=True)

    with rec_col:
        st.markdown("#### 💡 Personalised Recommendations")

        recs = []
        if backlogs > 0:
            recs.append(f"⚠️ Clear your {backlogs} academic backlog(s) — top priority for recruiters")
        if cgpa < 7.0:
            recs.append("📈 Improve your CGPA — aim for 7.5+ for competitive placement")
        if coding_score < 70:
            recs.append("💻 Strengthen coding skills — practice LeetCode & HackerRank daily")
        if mock_score < 65:
            recs.append("🎤 Practice mock interviews — score 75+ to stand out")
        if internships == 0:
            recs.append("🏢 Complete at least 1 internship to strengthen your resume")
        if hackathons == 0:
            recs.append("🏆 Participate in hackathons — great signal for tech recruiters")
        if projects_count < 3:
            recs.append("🔧 Build more projects — target 3–5 portfolio projects minimum")
        if comm_score < 70:
            recs.append("🗣️ Improve communication — join debate clubs or Toastmasters")
        if certifications < 2:
            recs.append("📜 Earn relevant certifications (AWS, Google, Coursera)")
        if github_repos < 5:
            recs.append("🐙 Increase GitHub activity — recruiters check your repositories")
        if aptitude_score < 65:
            recs.append("🧮 Improve aptitude score — practice quantitative & verbal reasoning")
        if attendance < 75:
            recs.append("📋 Improve attendance — many companies check academic regularity")
        if linkedin_conn < 150:
            recs.append("🔗 Grow your LinkedIn network to 150+ connections")

        if not recs:
            recs.append("⭐ Excellent profile! Maintain your current standards.")
            recs.append("🌟 Consider advanced certifications to further differentiate.")
            recs.append("🚀 Explore leadership roles in clubs or open-source projects.")

        for rec in recs:
            st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)
