
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MedVision AI | X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1525;
    color: #F1F5F9;
}

header {
    background-color: transparent !important;
}

[data-testid="stSidebar"] {
    background-color: #151F33;
    border-right: 1px solid #26344D;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 5px;
}

.subtitle {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 15px;
}

.info-card {
    background: #18243A;
    border: 1px solid #2B3A55;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #173A48, #18243A);
    border: 1px solid #267B83;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.result-label {
    color: #94A3B8;
    font-size: 14px;
}

.result-value {
    color: #5EEAD4;
    font-size: 30px;
    font-weight: 800;
}

.small-label {
    color: #94A3B8;
    font-size: 13px;
}

div.stButton > button {
    background: #0D9488;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #0F766E;
    color: white;
    border: none;
}

hr {
    border-color: #26344D;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras", compile=False)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🫁 MedVision AI")

    st.caption("INTELLIGENT MEDICAL IMAGING")

    st.divider()

    st.markdown("### Navigation")

    st.markdown("🔍 X-Ray Classification")

    st.markdown("📊 Model Information")

    st.divider()

    st.markdown("### Model Details")

    st.markdown("""
    - **Architecture:** Custom CNN
    - **Framework:** TensorFlow
    - **Input:** 250 × 300 × 3
    - **Classes:** 3
    """)

    st.divider()

    st.caption("MedVision AI | Educational Project")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Chest X-Ray Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered classification for COVID-19, Normal and Pneumonia chest X-rays.</div>',
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# MAIN LAYOUT
# ==========================================

left_col, right_col = st.columns([1.1, 0.9], gap="large")


# ==========================================
# LEFT: IMAGE UPLOAD
# ==========================================

with left_col:

    st.markdown(
        '<div class="section-title">📤 Upload X-Ray Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )

        st.markdown(
            f'<div class="small-label">File: {uploaded_file.name}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown("""
        <div class="info-card" style="text-align:center; padding:65px 20px;">
            <div style="font-size:55px;">🩻</div>
            <h3 style="color:#E2E8F0;">No Image Uploaded</h3>
            <p style="color:#94A3B8;">
                Upload a chest X-ray image to start the AI analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


    predict_button = st.button(
        "🔍 Analyze X-Ray",
        type="primary",
        disabled=uploaded_file is None
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button and uploaded_file:

    with st.spinner("AI is analyzing the X-ray..."):

        # Preprocessing
        img = image.resize((300, 250))

        img_array = np.array(img, dtype=np.float32)

        img_array = np.expand_dims(img_array, axis=0)

        # If you normalized your training images using /255,
        # uncomment the next line:
        # img_array = img_array / 255.0

        # Prediction
        predictions = model.predict(img_array, verbose=0)[0]

        class_names = ["COVID", "Normal", "Pneumonia"]

        predicted_index = np.argmax(predictions)

        predicted_class = class_names[predicted_index]

        confidence = float(predictions[predicted_index]) * 100


    # ==========================================
    # RIGHT: RESULTS
    # ==========================================

    with right_col:

        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED CLASS</div>
            <div class="result-value">{predicted_class}</div>
            <br>
            <div class="result-label">Model Confidence</div>
            <h2 style="color:#FFFFFF;">{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("### Class Probabilities")

        for i, name in enumerate(class_names):

            probability = float(predictions[i])

            st.markdown(
                f"**{name}**"
                f"<span style='float:right;color:#5EEAD4;'>"
                f"{probability*100:.2f}%</span>",
                unsafe_allow_html=True
            )

            st.progress(probability)

        st.write("")

        # Highest and lowest confidence classes
        st.markdown("### Prediction Summary")

        st.markdown(f"""
        <div class="info-card">
            <p class="small-label">Most likely classification</p>
            <h3 style="color:#5EEAD4;">{predicted_class}</h3>
            <p class="small-label">
                The model assigned the highest softmax score
                to this class.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Educational demonstration only. Not intended for clinical diagnosis."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center style='color:#64748B;'>"
    "MedVision AI | Powered by TensorFlow & Streamlit"
    "</center>",
    unsafe_allow_html=True
)
