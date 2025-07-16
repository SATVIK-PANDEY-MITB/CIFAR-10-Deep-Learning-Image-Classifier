from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ------------------- App Configuration -------------------
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="📷",
    layout="centered",
)

# ------------------- Custom Styling -------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            color: white;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            border: none;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stFileUploader label {
            font-size: 18px;
        }
        .prediction-box {
            background-color: #e8f0fe;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
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

# ------------------- Main Title -------------------
st.title("🎯 CIFAR-10 Image Classifier")
st.markdown("Upload an image of size **32x32**, and the model will predict its class.")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose a 32x32 RGB image", type=["png", "jpg", "jpeg"])

# ------------------- Image Classification -------------------
if uploaded_file:
    with st.spinner("🔍 Analyzing the image..."):
        image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
        st.image(image, caption='✅ Uploaded Image', use_container_width=True)

        img_array = np.array(image) / 255.0
        img_array = img_array.astype(np.float32)

        if img_array.shape == (32, 32, 3):
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = np.max(prediction) * 100

            # Top-3 Predictions
            top3_indices = prediction[0].argsort()[-3:][::-1]
            top3_classes = [(class_names[i], prediction[0][i] * 100) for i in top3_indices]

            st.markdown(f"""<div class="prediction-box">
                            <h3>🧠 Prediction: <span style='color:#3366cc'>{predicted_class.upper()}</span></h3>
                            <p>📈 Confidence: <b>{confidence:.2f}%</b></p>
                            </div>""", unsafe_allow_html=True)

            st.markdown("### 🔝 Top 3 Predictions")
            for i, (cls, conf) in enumerate(top3_classes, 1):
                st.write(f"**{i}. {cls.capitalize()}** — {conf:.2f}%")
        else:
            st.error("❌ Uploaded image must be a 32x32 RGB image.")

# ------------------- Footer / Info -------------------
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Model**: Transfer learning CNN trained on CIFAR-10
    - **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
    - **Created with**: Streamlit + TensorFlow + PIL + NumPy
    - 💡 Tip: Use online tools to resize your image to **32x32** before uploading.
    """)

