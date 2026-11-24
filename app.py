
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Vision | Dog vs Cat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - CHATGPT INSPIRED
# ==========================================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f7f7f8;
    color: #202123;
}

/* Hide Streamlit Header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #202123;
    border-right: 1px solid #343541;
}

section[data-testid="stSidebar"] * {
    color: #ececf1 !important;
}

/* Main Content */
.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Title */
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    color: #202123;
    letter-spacing: -1.5px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #777;
    margin-bottom: 35px;
}

/* Cards */
.custom-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e5e5;
    margin-top: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background-color: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px dashed #b8b8c0;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    background-color: #10a37f;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #0d8c6d;
    color: white;
    border: none;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #f0fdf9;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #d1fae5;
}

/* Image */
[data-testid="stImage"] img {
    border-radius: 15px;
}

/* Divider */
hr {
    border-color: #e5e5e5;
}

/* Info Text */
.small-text {
    color: #888;
    font-size: 13px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.markdown("# 🤖 AI Vision")

    st.markdown("---")

    st.markdown("### 🐶 Dog vs Cat")

    st.caption("AI Image Classification")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "An AI-powered image classification "
        "application using a Convolutional "
        "Neural Network (CNN)."
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Architecture:** CNN")

    st.write("**Input Shape:** 256 × 256 × 3")

    st.write("**Output:** Binary Classification")

    st.markdown("---")

    st.caption("Powered by TensorFlow & Streamlit")


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "dog_cat_model.keras"
    )


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# MAIN HEADER
# ==========================================
st.markdown(
    '<div class="main-title">AI Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent assistant for image classification'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# WELCOME MESSAGE
# ==========================================
st.markdown("""
<div class="custom-card">

<h3 style="margin-top:0; color:#202123;">
👋 Welcome to AI Vision!
</h3>

<p style="color:#666; font-size:16px;">
Upload an image of a dog or cat, and I'll analyze it
using my trained Convolutional Neural Network.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# IMAGE UPLOADER
# ==========================================
st.markdown("### 📤 Upload your image")

uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ==========================================
# PREDICTION HISTORY
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# IMAGE PREVIEW & PREDICTION
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼️ Image Preview")

    st.image(
        image,
        use_container_width=True
    )

    st.write("")

    if st.button("✨ Analyze Image", use_container_width=True):

        with st.spinner("AI is analyzing your image..."):

            # Preprocessing
            img = image.resize((256, 256))

            img_array = np.array(img).astype("float32")

            # Normalize
            img_array = img_array / 255.0

            # Batch Dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(
                img_array,
                verbose=0
            )

            probability = float(prediction[0][0])

            # Classification
            if probability >= 0.5:

                animal = "Dog 🐶"
                confidence = probability * 100

            else:

                animal = "Cat 🐱"
                confidence = (1 - probability) * 100

        # Save History
        st.session_state.history.append({
            "animal": animal,
            "confidence": confidence
        })

        # ==================================
        # RESULT
        # ==================================
        st.markdown("---")

        st.markdown("## 🤖 AI Analysis Result")

        if "Dog" in animal:
            result_color = "#e0f2fe"
            text_color = "#0369a1"
        else:
            result_color = "#fef3c7"
            text_color = "#92400e"

        st.markdown(
            f"""
            <div style="
                background:{result_color};
                padding:30px;
                border-radius:18px;
                text-align:center;
                border:1px solid #e5e5e5;
            ">
                <p style="color:{text_color}; font-size:17px;">
                    The AI predicts that this image contains
                </p>

                <h1 style="
                    color:{text_color};
                    font-size:42px;
                    margin:10px;
                ">
                    {animal}
                </h1>

                <p style="color:{text_color};">
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Confidence Metric
        st.metric(
            "🎯 Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.caption(
            "Confidence represents the predicted probability "
            "of the selected class, not a guarantee of correctness."
        )


# ==========================================
# PREDICTION HISTORY
# ==========================================
if st.session_state.history:

    st.markdown("---")

    st.markdown("### 📚 Recent Predictions")

    for item in reversed(st.session_state.history[-5:]):

        st.markdown(
            f"""
            <div style="
                background:white;
                padding:15px 20px;
                border-radius:12px;
                border:1px solid #e5e5e5;
                margin-bottom:10px;
            ">
                <b style="color:#202123;">
                    {item['animal']}
                </b>

                <span style="
                    float:right;
                    color:#10a37f;
                    font-weight:600;
                ">
                    {item['confidence']:.2f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown(
    '<p class="small-text">'
    'AI Vision · Built with TensorFlow & Streamlit'
    '</p>',
    unsafe_allow_html=True
)
