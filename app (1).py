import streamlit as st
import pandas as pd
import pickle

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── Load Model & Artifacts ───────────────────────────────────────────────────
model           = pickle.load(open("model.pkl", "rb"))
encoders        = pickle.load(open("encoders.pkl", "rb"))
feature_columns = pickle.load(open("columns.pkl", "rb"))
target_encoder  = pickle.load(open("target_encoder.pkl", "rb"))

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Buttons */
    .stButton>button {
        background-color: #1abc9c !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5em 1.2em !important;
        font-size: 1rem !important;
        transition: background 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #16a085 !important;
    }
    /* Sliders */
    input[type=range] {
        accent-color: #1abc9c;
    }
    /* Labels */
    label, .stRadio>label {
        font-weight: 600;
        color: #0d3b66;
    }
    </style>
""", unsafe_allow_html=True)

# ─── App Header ──────────────────────────────────────────────────────────────
st.title("💼 Employee Salary Prediction")
st.write("Predict if an employee earns **>50K** or **≤50K** based on key details.")

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
st.sidebar.header("🔧 Employee Details")

def encode_row(df: pd.DataFrame) -> pd.DataFrame:
    """Apply each encoder to its column in the DataFrame."""
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = enc.transform(df[col])
    return df

# Essential features only:
age            = st.sidebar.slider("Age", 18, 65, 30)
workclass      = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
education      = st.sidebar.selectbox("Education", encoders['education'].classes_)
occupation     = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
gender         = st.sidebar.radio("Gender", encoders['gender'].classes_)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
capital_gain   = st.sidebar.number_input("Capital Gain", min_value=0, step=100, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)

# Build input row
input_dict = {
    'age': age,
    'workclass': workclass,
    'education': education,
    'occupation': occupation,
    'gender': gender,
    'marital-status': marital_status,
    'capital-gain': capital_gain,
    'hours-per-week': hours_per_week
}
input_df = pd.DataFrame([input_dict])

# Add any missing model columns with default 0
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to model’s expectation
input_df = input_df[feature_columns]

# Encode categorical columns
encoded_input = encode_row(input_df.copy())

# ─── Input Preview ──────────────────────────────────────────────────────────
st.markdown("### 🔍 Input Preview")
st.dataframe(input_df, use_container_width=True)

# ─── Prediction ──────────────────────────────────────────────────────────────
if st.button("🚀 Predict Salary"):
    pred  = model.predict(encoded_input)[0]
    prob  = model.predict_proba(encoded_input)[0, pred]
    label = target_encoder.inverse_transform([pred])[0]

    st.success(f"🧾 Prediction: **{label}**")
    st.progress(int(prob * 100))
    st.info(f"Confidence: {prob:.1%}")

# ─── Batch Prediction ────────────────────────────────────────────────────────
st.markdown("### 📂 Batch Prediction")
batch_file = st.file_uploader("Upload CSV (same columns as preview)", type=['csv'])
if batch_file:
    df_batch = pd.read_csv(batch_file)
    df_enc   = encode_row(df_batch.copy())

    preds  = model.predict(df_enc)
    labels = target_encoder.inverse_transform(preds)
    df_batch['Predicted Income'] = labels

    st.dataframe(df_batch.head(), use_container_width=True)
    st.download_button(
        "⬇ Download Full Results",
        df_batch.to_csv(index=False),
        file_name="salary_predictions.csv",
        mime="text/csv"
    )
