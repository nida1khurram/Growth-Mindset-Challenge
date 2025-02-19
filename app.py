# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import cv2
# from PIL import Image
# import io

# # Load the pre-trained model (MobileNetV2 for image classification)
# model = MobileNetV2(weights="imagenet")

# # Streamlit App
# st.title("🖼️ Image Processing & Classification App")
# st.write("Upload an image to classify it, apply filters, resize, convert formats, and download.")

# # File uploader
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="📸 Uploaded Image", use_column_width=True)

#     # Convert image to OpenCV format (BGR)
#     img_array = np.array(img)

#     # 📌 Image Classification
#     if st.button("Classify Image 🧠"):
#         st.subheader("🔍 Image Classification Results")
        
        

#     # 📌 Image Processing Filters
#     st.subheader("🎨 Apply Image Filters")
#     filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Blur", "Edge Detection"])

#     processed_img = img_array  # Default is original image

#     if filter_option == "Grayscale":
#         processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         st.image(processed_img, caption="🖤 Grayscale Image", use_column_width=True)
#         final_image = Image.fromarray(processed_img).convert("L")  # Convert to grayscale PIL Image

#     elif filter_option == "Blur":
#         processed_img = cv2.GaussianBlur(img_array, (15, 15), 0)
#         st.image(processed_img, caption="🌫️ Blurred Image", use_column_width=True)
#         final_image = Image.fromarray(processed_img)

#     elif filter_option == "Edge Detection":
#         gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         processed_img = cv2.Canny(gray_img, 100, 200)
#         st.image(processed_img, caption="⚡ Edge Detection Image", use_column_width=True)
#         final_image = Image.fromarray(processed_img).convert("L")  # Convert to grayscale PIL Image

#     else:
#         st.image(img, caption="📸 Original Image", use_column_width=True)
#         final_image = img

    

#     # 📌 Format Conversion & Download Button
#     st.subheader("📥 Download Processed Image")
#     download_format = st.selectbox("Select download format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])

#     # Convert image to bytes
#     img_bytes_io = io.BytesIO()  # Corrected
#     final_image.save(img_bytes_io, format=download_format)
#     img_bytes_io.seek(0)  # Reset buffer position

#     # Download button
#     st.download_button(
#         label=f"⬇️ Download as {download_format}",
#         data=img_bytes_io,
#         file_name=f"processed_image.{download_format.lower()}",
#         mime=f"image/{download_format.lower()}",
#     )

import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import io

# Streamlit App
st.set_page_config(page_title="Image Processing App", layout="centered")

# Custom CSS to beautify the UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1e3a8a;
        font-size: 2.5rem !important;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 0.5rem;
    }
    .stSelectbox {
        border-radius: 0.5rem;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1e3a8a;
        color: white;
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Image Processing App")
st.write("Upload an image to apply filters, convert formats, and download.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    
    # Resize image to a smaller size
    max_size = (400, 400)
    img.thumbnail(max_size)
    
    st.image(img, caption="📸 Uploaded Image (Resized)", use_column_width=True)

    # 📌 Image Processing Filters
    st.subheader("🎨 Apply Image Filters")
    filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Blur", "Edge Detection"])

    if filter_option == "Grayscale":
        processed_img = ImageOps.grayscale(img)
        st.image(processed_img, caption="🖤 Grayscale Image", use_column_width=True)
        final_image = processed_img

    elif filter_option == "Blur":
        processed_img = img.filter(ImageFilter.GaussianBlur(radius=5))
        st.image(processed_img, caption="🌫️ Blurred Image", use_column_width=True)
        final_image = processed_img

    elif filter_option == "Edge Detection":
        processed_img = img.filter(ImageFilter.FIND_EDGES)
        st.image(processed_img, caption="⚡ Edge Detection Image", use_column_width=True)
        final_image = processed_img

    else:
        st.image(img, caption="📸 Original Image", use_column_width=True)
        final_image = img

    # 📌 Format Conversion & Download Button
    st.subheader("📥 Download Processed Image")
    download_format = st.selectbox("Select download format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])

    # Convert image to bytes
    img_bytes_io = io.BytesIO()
    final_image.save(img_bytes_io, format=download_format)
    img_bytes_io.seek(0)  # Reset buffer position

    # Download button
    st.download_button(
        label=f"⬇️ Download as {download_format}",
        data=img_bytes_io,
        file_name=f"processed_image.{download_format.lower()}",
        mime=f"image/{download_format.lower()}",
    )

# Footer
st.markdown(
    """
    <div class="footer">
        Created by Nida | 
        <a href="https://www.linkedin.com/in/nida-khurram/"  target="_blank" style="color: white;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

