import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/final_model.keras"
    )

model = load_model()

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️"
)

st.title("CIFAR-10 Image Classifier")

st.write(
    "Upload an image and the trained CNN "
    "will predict its CIFAR-10 class."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    resized_image = image.resize(
        (32, 32)
    )

    image_array = np.asarray(
        resized_image
    ).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    st.subheader(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    st.subheader("Class Probabilities")

    probability_data = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(
        probability_data
    )