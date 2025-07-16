from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ------------------- App Configuration -------------------
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🤖",
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

# ------------------- Page Content -------------------
st.markdown("<div class='title'>🖼️ CIFAR-10 Image Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a 32x32 image of an object, and get its predicted category instantly!</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a 32x32 image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], label_visibility="visible")

# ------------------- Image Classification -------------------
if uploaded_file:
    with st.spinner("🔍 Analyzing your image..."):
        image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
        st.image(image, caption='✅ Uploaded Image', use_container_width=True)

        img_array = np.array(image) / 255.0
        img_array = img_array.astype(np.float32)

        if img_array.shape == (32, 32, 3):
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = np.max(prediction) * 100

            st.markdown(f"""<div class="prediction-box">
                            <div class="prediction-text">🧠 Prediction: {predicted_class.upper()}</div>
                            <div class="confidence-text">📈 Confidence: {confidence:.2f}%</div>
                            </div>""", unsafe_allow_html=True)
        else:
            st.error("❌ Uploaded image must be a 32x32 RGB image.")

# ------------------- Footer / Info -------------------
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Model**: CNN with transfer learning trained on CIFAR-10
    - **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
    - **Tech**: Streamlit, TensorFlow, Pillow
    - 💡 Tip: Resize your image to **32x32 pixels** before uploading for accurate results.
    """)

