import streamlit as st
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("cat_dog_model.keras")

st.title("🐱 Cat vs Dog Classifier")

st.write("Upload an image and the CNN model will classify it as a Cat or Dog.")


# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(img, caption="Uploaded Image", width=400)
    if st.button("Predict"):

        img = img.resize((180, 180))

        # Convert image to numpy array
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)


        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]


        # Classify
        if prediction > 0.5:
            label = "Dog"
            confidence = prediction
        else:
            label = "Cat"
            confidence = 1 - prediction


        # Display result
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")