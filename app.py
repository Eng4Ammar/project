import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MODEL_PATH = "model.keras"
IMG_WIDTH = 300
IMG_HEIGHT = 250
CLASS_NAMES = ["COVID", "Normal", "Pneumonia"]

st.set_page_config(
    page_title="MedVision AI | X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS
# ==========================================
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
""",
    unsafe_allow_html=True,
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
  return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
  # Fix EXIF orientation if present and ensure 3-channel RGB
  oriented = ImageOps.exif_transpose(image).convert("RGB")
  resized = oriented.resize((IMG_WIDTH, IMG_HEIGHT))
  img_array = np.asarray(resized, dtype=np.float32) / 255.0
  return np.expand_dims(img_array, axis=0)


def predict_xray(model, image: Image.Image) -> np.ndarray:
  input_tensor = preprocess_image(image)
  raw_preds = model.predict(input_tensor, verbose=0)[0]
  return np.asarray(raw_preds, dtype=np.float32)


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
  st.markdown(f"""
    - **Architecture:** Custom CNN
    - **Framework:** TensorFlow
    - **Input:** {IMG_HEIGHT} × {IMG_WIDTH} × 3
    - **Classes:** {len(CLASS_NAMES)}
    """)
  st.divider()

  st.caption("MedVision AI | Educational Project")


# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">Chest X-Ray Analysis</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered classification for COVID-19, Normal and'
    ' Pneumonia chest X-rays.</div>',
    unsafe_allow_html=True,
)


# ==========================================
# LOAD MODEL
# ==========================================
try:
  model = load_model(MODEL_PATH)
except Exception as e:
  st.error(f"Unable to load model: {e}")
  st.stop()


# ==========================================
# MAIN LAYOUT
# ==========================================
left_col, right_col = st.columns([1.1, 0.9], gap="large")
image = None


# ==========================================
# LEFT: IMAGE UPLOAD
# ==========================================
with left_col:
  st.markdown(
      '<div class="section-title">📤 Upload X-Ray Image</div>',
      unsafe_allow_html=True,
  )

  uploaded_file = st.file_uploader(
      "Choose a chest X-ray image",
      type=["jpg", "jpeg", "png"],
      label_visibility="collapsed",
  )

  if uploaded_file is not None:
    try:
      with Image.open(uploaded_file) as opened_img:
        image = ImageOps.exif_transpose(opened_img).convert("RGB")

      st.image(
          image, caption="Uploaded Chest X-Ray", use_container_width=True
      )
      st.markdown(
          f'<div class="small-label">File: {uploaded_file.name}</div>',
          unsafe_allow_html=True,
      )
    except Exception as e:
      image = None
      st.error(f"Could not read the uploaded image: {e}")
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


# ==========================================
# PREDICTION & RESULTS
# ==========================================
if predict_button and image is not None:
  try:
    with st.spinner("AI is analyzing the X-ray..."):
      predictions = predict_xray(model, image)

    if predictions.shape[0] != len(CLASS_NAMES):
      st.error(
          f"Model output shape ({predictions.shape[0]}) does not match expected"
          f" classes ({len(CLASS_NAMES)})."
      )
      st.stop()

    predicted_index = int(np.argmax(predictions))
    lowest_index = int(np.argmin(predictions))

    predicted_class = CLASS_NAMES[predicted_index]
    lowest_class = CLASS_NAMES[lowest_index]

    confidence = float(predictions[predicted_index]) * 100.0
    lowest_confidence = float(predictions[lowest_index]) * 100.0

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
      st.markdown("### Class Probabilities")

      for i, name in enumerate(CLASS_NAMES):
        probability = max(0.0, min(1.0, float(predictions[i])))

        st.markdown(
            f"**{name}**"
            "<span style='float:right;color:#5EEAD4;'>"
            f"{probability * 100:.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.progress(probability)

      st.write("")
      st.markdown("### Prediction Summary")

      st.markdown(
          f"""
            <div class="info-card">
                <p class="small-label">Most likely classification</p>
                <h3 style="color:#5EEAD4;">{predicted_class} ({confidence:.2f}%)</h3>
                <p class="small-label">
                    Least likely class: <strong>{lowest_class}</strong> ({lowest_confidence:.2f}%).
                    The model assigned the highest softmax score to the predicted class.
                </p>
            </div>
            """,
          unsafe_allow_html=True,
      )

      st.caption(
          "Educational demonstration only. Not intended for clinical diagnosis."
      )

  except Exception as e:
    st.error(f"An error occurred during analysis: {e}")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown(
    "<center style='color:#64748B;'>MedVision AI | Powered by TensorFlow &"
    " Streamlit</center>",
    unsafe_allow_html=True,
)
