import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
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
    "truck",
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/final_model.keras")


model = load_model()

st.set_page_config(
    page_title="CIFAR-10 image classifier",
    page_icon=":material/image_search:",
    layout="centered",
)

with st.container(horizontal_alignment="center"):
    st.title("CIFAR-10 image classifier")
    st.caption(
        "Upload a photo and a trained CNN will guess its class "
        "among 10 CIFAR-10 categories."
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info(
        "Upload a JPG or PNG image to get started.",
        icon=":material/upload:",
    )
else:
    image = Image.open(uploaded_file).convert("RGB")

    resized_image = image.resize((32, 32))
    image_array = np.asarray(resized_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    with st.spinner("Classifying image..."):
        probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    image_col, result_col = st.columns(2, vertical_alignment="center")

    with image_col:
        st.image(image, caption="Uploaded image", width="stretch")

    with result_col:
        with st.container(border=True):
            st.metric(
                "Prediction",
                predicted_class.capitalize(),
                border=False,
            )
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                border=False,
            )

    st.subheader("Class probabilities")

    probability_df = (
        pd.DataFrame(
            {
                "class": CLASS_NAMES,
                "probability": probabilities,
            }
        )
        .sort_values("probability", ascending=False)
        .set_index("class")
    )

    st.dataframe(
        probability_df,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Probability",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            )
        },
        width="stretch",
    )