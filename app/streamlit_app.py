from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('TransferlearningSP.keras')

model = load_model()

st.title("CIFAR-10 Image Classifier")
uploaded_file = st.file_uploader("Upload a 32x32 image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))  # Force RGB + resize
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = img_array.astype(np.float32)

    if img_array.shape == (32, 32, 3):
        prediction = model.predict(np.expand_dims(img_array, axis=0))  # Shape: (1, 32, 32, 3)
        predicted_class = class_names[np.argmax(prediction)]
        st.subheader(f"Prediction: {predicted_class}")
    else:
        st.error("Uploaded image format is not supported. Please upload a 32x32 RGB image.")
