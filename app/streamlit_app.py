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

if uploaded_file is not None:
    # Open and convert to RGB
    image = Image.open(uploaded_file).convert('RGB')
    
    # Resize to 32x32 for CIFAR-10
    image = image.resize((32, 32))
    
    st.image(image, caption="Uploaded Image", use_column_width=False)
    
    # Preprocess image
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: {predicted_class}")
