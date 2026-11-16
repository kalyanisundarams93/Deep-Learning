import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection")
st.write("Detect whether a person is wearing a face mask.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("facemask.keras")

model = load_model()

# Class names from your training dataset
class_names = ['with_mask', 'without_mask']

# Camera input
image = st.camera_input("Capture your face")

if image is not None:

    # Read image
    img = Image.open(image).convert("RGB")

    # Resize image
    img = img.resize((256, 256))

    # Convert image into array
    img_array = np.array(img)

    # Preprocessing
    img_array = img_array.astype("float32")
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction[0])

    label = class_names[class_index]

    confidence = prediction[0][class_index] * 100

    # Display result
    st.subheader("Prediction Result")

    if label == "with_mask":
        st.success(f"With Mask 😷")
    else:
        st.error("Without Mask")

    st.write(f"Confidence: {confidence:.2f}%")
