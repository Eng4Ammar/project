from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "dog_cat_model.keras"
IMAGE_SIZE = (256, 256)

st.set_page_config(
    page_title="AI Vision | Dog vs Cat",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f8;
        color: #202123;
    }
    section[data-testid="stSidebar"] {
        background-color: #202123;
        border-right: 1px solid #343541;
    }
    section[data-testid="stSidebar"] * {
        color: #ececf1 !important;
    }
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        color: #202123;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #777;
        margin-bottom: 35px;
    }
    .custom-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e5e5;
        margin-top: 20px;
    }
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px dashed #b8b8c0;
    }
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        background-color: #10a37f;
        color: white;
        border: none;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #0d8c6d;
        color: white;
    }
    [data-testid="stMetric"] {
        background: #f0fdf9;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #d1fae5;
    }
    hr {
        border-color: #e5e5e5;
    }
    .small-text {
        color: #888;
        font-size: 13px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def preprocess_image(image):
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(pixels, axis=0)


def validate_model(model):
    input_shape = tuple(model.input_shape)
    if input_shape[1:] != (256, 256, 3):
        raise ValueError(
            f"Expected model input (None, 256, 256, 3), got {input_shape}."
        )

    output_shape = tuple(model.output_shape)
    if output_shape[-1] != 1:
        raise ValueError(
            f"Expected one sigmoid output, got model output shape {output_shape}."
        )


with st.sidebar:
    st.title("🤖 AI Vision")
    st.divider()
    st.subheader("🐶 Dog vs Cat")
    st.caption("AI Image Classification")
    st.divider()
    st.subheader("About")
    st.write(
        "An AI-powered application that classifies uploaded images as "
        "dog or cat using a trained Convolutional Neural Network."
    )
    st.divider()
    st.subheader("Model Information")
    st.write("**Architecture:** Custom CNN")
    st.write("**Input:** 256 × 256 × 3")
    st.write("**Output:** Binary classification")
    st.write("**Framework:** TensorFlow")
    st.divider()
    st.caption("Powered by TensorFlow & Streamlit")

st.markdown(
    '<div class="main-title">AI Vision</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Your intelligent assistant for dog and cat image classification"
    "</div>",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

if not MODEL_PATH.is_file():
    st.error(f"Model file was not found: {MODEL_PATH.name}")
    st.stop()

try:
    model = load_model()
    validate_model(model)
except (OSError, ValueError, TypeError, ImportError, tf.errors.OpError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

st.markdown(
    """
    <div class="custom-card">
    <h3 style="margin-top:0;color:#202123;">👋 Welcome to AI Vision!</h3>
    <p style="color:#666;font-size:16px;">
    Upload an image of a dog or cat and the trained model will classify it.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 📤 Upload your image")
uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=("jpg", "jpeg", "png"),
    label_visibility="collapsed",
)

if uploaded_file is not None:
    try:
        with Image.open(BytesIO(uploaded_file.getvalue())) as opened:
            image = opened.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ) as error:
        st.error(f"Could not read the uploaded image: {error}")
        st.stop()

    st.markdown("### 🖼️ Image Preview")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("✨ Analyze Image", use_container_width=True):
        with st.spinner("AI is analyzing your image..."):
            try:
                result = model.predict(preprocess_image(image), verbose=0)
                probability = float(np.asarray(result).reshape(-1)[0])
            except (ValueError, tf.errors.InvalidArgumentError) as error:
                st.error(f"Could not analyze the image: {error}")
                st.stop()

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            st.error("The model returned an invalid probability.")
            st.stop()

        if probability >= 0.5:
            animal = "Dog 🐶"
            confidence = probability * 100
            result_color = "#e0f2fe"
            text_color = "#0369a1"
        else:
            animal = "Cat 🐱"
            confidence = (1 - probability) * 100
            result_color = "#fef3c7"
            text_color = "#92400e"

        st.session_state.history.append(
            {"animal": animal, "confidence": confidence}
        )

        st.divider()
        st.markdown("## 🤖 AI Analysis Result")
        st.markdown(
            f"""
            <div style="
                background:{result_color};
                padding:30px;
                border-radius:18px;
                text-align:center;
                border:1px solid #e5e5e5;">
                <p style="color:{text_color};font-size:17px;">
                    The AI predicts that this image contains
                </p>
                <h1 style="color:{text_color};font-size:42px;margin:10px;">
                    {animal}
                </h1>
                <p style="color:{text_color};">
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("🎯 Prediction Confidence", f"{confidence:.2f}%")
        st.progress(max(0.0, min(1.0, confidence / 100)))
        st.caption(
            "Confidence is the model probability for the selected class "
            "and is not a guarantee of correctness."
        )

if st.session_state.history:
    st.divider()
    st.subheader("📚 Recent Predictions")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(
            f"""
            <div style="
                background:white;
                padding:15px 20px;
                border-radius:12px;
                border:1px solid #e5e5e5;
                margin-bottom:10px;">
                <b style="color:#202123;">{item['animal']}</b>
                <span style="float:right;color:#10a37f;font-weight:600;">
                    {item['confidence']:.2f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    '<p class="small-text">AI Vision · Built with TensorFlow & Streamlit</p>',
    unsafe_allow_html=True,
)
