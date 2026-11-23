from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.keras"
IMAGE_SIZE = (300, 250)
INPUT_SHAPE = (250, 300, 3)
CLASS_NAMES = ("COVID", "Normal", "Pneumonia")

st.set_page_config(
    page_title="MedVision AI | X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def preprocess_image(image):
    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def validate_model(model):
    input_shape = tuple(model.input_shape)
    if input_shape[1:] != INPUT_SHAPE:
        raise ValueError(
            f"Expected model input (None, {INPUT_SHAPE}), got {input_shape}."
        )

    output_shape = tuple(model.output_shape)
    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, got {output_shape[-1]}."
        )


with st.sidebar:
    st.title("🫁 MedVision AI")
    st.caption("INTELLIGENT MEDICAL IMAGING")
    st.divider()
    st.subheader("Navigation")
    st.markdown("🔍 X-Ray Classification")
    st.markdown("📊 Model Information")
    st.divider()
    st.subheader("Model Details")
    st.markdown(
        f"""
        - **Architecture:** Custom CNN
        - **Framework:** TensorFlow {tf.__version__}
        - **Input:** {INPUT_SHAPE[0]} × {INPUT_SHAPE[1]} × {INPUT_SHAPE[2]}
        - **Classes:** {len(CLASS_NAMES)}
        """
    )
    st.divider()
    st.caption("Educational project only")

st.markdown(
    '<div class="main-title">Chest X-Ray Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered classification for COVID-19, Normal, and Pneumonia chest X-rays.</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(f"Model file was not found: {MODEL_PATH.name}")
    st.stop()

try:
    model = load_model()
    validate_model(model)
except (OSError, ValueError, TypeError, ImportError, tf.errors.OpError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown(
        '<div class="section-title">📤 Upload X-Ray Image</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "The saved model does not include class names. Results use the "
        "configured order: COVID, Normal, Pneumonia."
    )
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=("jpg", "jpeg", "png"),
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        try:
            with Image.open(BytesIO(uploaded_file.getvalue())) as opened_image:
                image = opened_image.convert("RGB")
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
        ) as error:
            st.error(f"Could not read the uploaded image: {error}")

    if image is not None:
        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            width="stretch",
        )
        st.caption(f"File: {uploaded_file.name}")
    else:
        st.markdown(
            """
            <div class="info-card" style="text-align:center; padding:65px 20px;">
                <div style="font-size:55px;">🩻</div>
                <h3 style="color:#E2E8F0;">No Image Uploaded</h3>
                <p style="color:#94A3B8;">
                    Upload a chest X-ray image to start the AI analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    predict_button = st.button(
        "🔍 Analyze X-Ray",
        type="primary",
        disabled=image is None,
    )

if predict_button and image is not None:
    with st.spinner("AI is analyzing the X-ray..."):
        try:
            batch = preprocess_image(image)
            raw_predictions = model.predict(batch, verbose=0)
            predictions = np.asarray(raw_predictions)
        except (ValueError, tf.errors.InvalidArgumentError) as error:
            st.error(f"Could not analyze the image: {error}")
            st.stop()

    if predictions.shape != (1, len(CLASS_NAMES)):
        st.error(
            "The model returned an unexpected result shape. "
            f"Expected (1, {len(CLASS_NAMES)}), got {predictions.shape}."
        )
        st.stop()

    scores = predictions[0]
    if not np.isfinite(scores).all():
        st.error("The model returned invalid probability values.")
        st.stop()

    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)
    ):
        st.error("The model did not return valid class probabilities.")
        st.stop()

    predicted_index = int(np.argmax(scores))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(scores[predicted_index]) * 100

    with right_col:
        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">PREDICTED CLASS</div>
                <div class="result-value">{predicted_class}</div>
                <br>
                <div class="result-label">Model Confidence</div>
                <h2 style="color:#FFFFFF;">{confidence:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.subheader("Class Probabilities")
        for name, score in zip(CLASS_NAMES, scores):
            probability = float(score)
            st.markdown(
                f"**{name}**"
                f"<span style='float:right;color:#5EEAD4;'>"
                f"{probability * 100:.2f}%</span>",
                unsafe_allow_html=True,
            )
            st.progress(probability)

        st.write("")
        st.subheader("Prediction Summary")
        st.markdown(
            f"""
            <div class="info-card">
                <p style="color:#94A3B8;">Most likely classification</p>
                <h3 style="color:#5EEAD4;">{predicted_class}</h3>
                <p style="color:#94A3B8;">
                    The model assigned the highest score to this class.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Educational demonstration only. Not intended for clinical diagnosis."
        )

st.divider()
st.markdown(
    "<center style='color:#64748B;'>"
    "MedVision AI | Powered by TensorFlow & Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
