import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Brain Tumor Detection")

# Load model
model = tf.keras.models.load_model(r'D:\Moaz\Fall_2025_2026\CS481\project_2\brain_tumor_cnn.keras')

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Image', use_container_width=True)
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    class_labels = ['no','yes']
    
    st.write(f"Prediction: {class_labels[class_idx]}")
    st.write(f"Confidence: {pred[0][class_idx]*100:.2f}%")