import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model('../models/model_vgg16_final.h5')

# Define the class names
class_names = ["Mild Demented", "Moderate Demented", "Non-Demented", "Very Mild Demented"]

# Function to load and preprocess the image
def load_and_preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Rescale
    return img_array

# Streamlit interface
st.set_page_config(page_title="Alzheimer's MRI Classification", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        background: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    .title {
        font-family: 'Helvetica', sans-serif;
        color: #4CAF50;
        text-align: center;
        padding: 20px 0;
    }
    .subtitle {
        font-family: 'Helvetica', sans-serif;
        color: #555;
        margin: 20px 0;
    }
    .section {
        padding: 20px 0;
    }
    .section img {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .prediction {
        font-family: 'Helvetica', sans-serif;
        color: #FF5722;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
    }
    .confidence {
        font-family: 'Helvetica', sans-serif;
        color: #2196F3;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 0.9em;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page content
st.markdown("<h1 class='title'>Alzheimer's MRI Classification</h1>", unsafe_allow_html=True)

# Introduction Section
st.markdown(
    """
    <div class='section'>
        <h2 class='subtitle'>Understanding Alzheimer's Disease</h2>
        <p class='subtitle'>Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and behavior. 
        Early detection and diagnosis are crucial for managing and treating the disease. 
        This application uses a deep learning model to classify MRI scans into one of four categories, helping in early detection and diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.image('images/img1.jpg', caption='Understanding Alzheimer’s Disease', use_column_width=False, width=600)

# Statistics Section
st.markdown(
    """
    <div class='section'>
        <h2 class='subtitle'>Alzheimer's Disease Statistics</h2>
        <p class='subtitle'>According to the Alzheimer's Association:</p>
        <ul class='subtitle'>
            <li>Over 6 million Americans are living with Alzheimer’s disease.</li>
            <li>By 2050, this number is projected to rise to nearly 13 million.</li>
            <li>Alzheimer's and other dementias cost the nation $355 billion annually.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)



st.image('images/stats.jpg', caption='Alzheimer’s Disease Statistics', use_column_width=False,width=600)

# How the Model Works Section
st.markdown(
    """
    <div class='section'>
        <h2 class='subtitle'>How the Model Works</h2>
        <p class='subtitle'>Our deep learning model is trained to analyze MRI scans and classify them into one of the following categories:</p>
        <ul class='subtitle'>
            <li>Mild Demented</li>
            <li>Moderate Demented</li>
            <li>Non-Demented</li>
            <li>Very Mild Demented</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
st.image('images/scan.png', caption='MRI Scans', use_column_width=False,width=600)
st.image('images/how_it_works.jpg', caption='How the Model Works', use_column_width=False,width=600)

# Upload and classify MRI image
st.markdown("<h2 class='subtitle'>Upload MRI Scan</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=False, width=300)
    st.write("Classifying...")

    # Preprocess the image
    img_array = load_and_preprocess_image(uploaded_file)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)

    # Display the results
    st.markdown(f"<p class='prediction'>Prediction: {class_names[predicted_class[0]]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence'>Confidence: {confidence[0]:.2f}</p>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class='footer'>
        <p>© 2024 Alzheimer's MRI Classification. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
