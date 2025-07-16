from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ------------------- App Configuration -------------------
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🖼️",
    layout="centered",
)

# ------------------- Custom Styling -------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .title {
            font-size: 40px;
            font-weight: 800;
            color: #ff4b4b;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            font-weight: 500;
            color: #444;
            text-align: center;
            margin-bottom: 30px;
        }
        .prediction-box {
            background-color: #fff3e0;
            border-left: 6px solid #ff6f00;
            padding: 1.5rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            text-align: center;
        }
        .prediction-text {
            font-size: 26px;
            font-weight: 700;
            color: #d84315;
        }
        .confidence-text {
            font-size: 18px;
            font-weight: 500;
            color: #444;
            margin-top: 8px;
        }
        .upload-label {
            font-weight: 600;
            font-size: 18px;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Load Model -------------------
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('TransferlearningSP.keras')

model = load_model()

# ------------------- Page Header -------------------
st.markdown("<div class='title'>🖼️ CIFAR-10 Image Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a 32x32 image of an object, and get its predicted category instantly!</div>", unsafe_allow_html=True)

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload a 32x32 image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], label_visibility="visible")

# ------------------- Classification -------------------
if uploaded_file:
    with st.spinner("🔍 Analyzing your image..."):
        # Load original image
        original_image = Image.open(uploaded_file).convert("RGB")

        # Resize for model prediction
        resized_image = original_image.resize((32, 32))

        # Upscale the resized image for clear display
        display_image = resized_image.resize((256, 256), resample=Image.NEAREST)

        # Show the crisp upscaled image (no deprecati

