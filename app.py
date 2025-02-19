import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import io

# Load the pre-trained model (MobileNetV2 for image classification)
model = MobileNetV2(weights="imagenet")

# Streamlit App
st.title("🖼️ Image Processing & Classification App")
st.write("Upload an image to classify it, apply filters, resize, convert formats, and download.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format (BGR)
    img_array = np.array(img)

    # 📌 Image Classification
    if st.button("Classify Image 🧠"):
        st.subheader("🔍 Image Classification Results")
        
        # Preprocess the image for MobileNetV2
        # img_resized = img.resize((224, 224))
        # img_array_resized = image.img_to_array(img_resized)
        # img_array_resized = np.expand_dims(img_array_resized, axis=0)
        # img_array_resized = preprocess_input(img_array_resized)

        # Predict class
        # preds = model.predict(img_array_resized)
        # decoded_preds = decode_predictions(preds, top=3)[0]

        # for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        #     st.write(f"**{i+1}. {label} ({score*100:.2f}%)**")

    # 📌 Image Processing Filters
    st.subheader("🎨 Apply Image Filters")
    filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Blur", "Edge Detection"])

    processed_img = img_array  # Default is original image

    if filter_option == "Grayscale":
        processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(processed_img, caption="🖤 Grayscale Image", use_column_width=True)
        final_image = Image.fromarray(processed_img).convert("L")  # Convert to grayscale PIL Image

    elif filter_option == "Blur":
        processed_img = cv2.GaussianBlur(img_array, (15, 15), 0)
        st.image(processed_img, caption="🌫️ Blurred Image", use_column_width=True)
        final_image = Image.fromarray(processed_img)

    elif filter_option == "Edge Detection":
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(gray_img, 100, 200)
        st.image(processed_img, caption="⚡ Edge Detection Image", use_column_width=True)
        final_image = Image.fromarray(processed_img).convert("L")  # Convert to grayscale PIL Image

    else:
        st.image(img, caption="📸 Original Image", use_column_width=True)
        final_image = img

    # 📌 Image Resizing Option
    # st.subheader("📏 Resize Image")
    # width = st.number_input("Width", value=img.width, min_value=1)
    # height = st.number_input("Height", value=img.height, min_value=1)
    # if st.button("Resize Image"):
    #     final_image = final_image.resize((width, height))
    #     st.image(final_image, caption="📏 Resized Image", use_column_width=True)

    # 📌 Format Conversion & Download Button
    st.subheader("📥 Download Processed Image")
    download_format = st.selectbox("Select download format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])

    # Convert image to bytes
    img_bytes_io = io.BytesIO()  # Corrected
    final_image.save(img_bytes_io, format=download_format)
    img_bytes_io.seek(0)  # Reset buffer position

    # Download button
    st.download_button(
        label=f"⬇️ Download as {download_format}",
        data=img_bytes_io,
        file_name=f"processed_image.{download_format.lower()}",
        mime=f"image/{download_format.lower()}",
    )
