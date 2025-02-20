
import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
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
    
    # Display original and processed images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📸 Original Image", use_container_width=True)

    # 📌 Image Processing Filters
    st.subheader("🎨 Apply Image Filters")
    filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Blur", "Edge Detection","Flip Horizontal", "Flip Vertical", "Rotate", "Sharpen"])

    if filter_option == "Grayscale":
        processed_img = ImageOps.grayscale(img)
        with col2:
            st.image(processed_img, caption="🖤 Grayscale Image", use_container_width=True)
        final_image = processed_img

    elif filter_option == "Blur":
        processed_img = img.filter(ImageFilter.GaussianBlur(radius=5))
        with col2:
            st.image(processed_img, caption="🌫️ Blurred Image", use_container_width=True)
        final_image = processed_img

    elif filter_option == "Edge Detection":
        gray_img = ImageOps.grayscale(img)
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        inverted_edges = ImageOps.invert(edges)
        with col2:
            st.image(inverted_edges, caption="⚡ Edge Detection (White Background)", use_container_width=True)
        final_image = inverted_edges

    elif filter_option == "Flip Horizontal":
        processed_img = ImageOps.mirror(img)
        with col2:
            st.image(processed_img, caption="↔️ Flipped Image (Horizontal)", use_container_width=True)
        final_image = processed_img

    elif filter_option == "Flip Vertical":
        processed_img = ImageOps.flip(img)
        with col2:
            st.image(processed_img, caption="↕️ Flipped Image (Vertical)", use_container_width=True)
        final_image = processed_img

    elif filter_option == "Rotate":
        rotation_angle = st.slider("Rotation Angle", 0, 360, 0)
        processed_img = img.rotate(rotation_angle)
        with col2:
            st.image(processed_img, caption=f"🔄 Rotated Image ({rotation_angle}°)", use_container_width=True)
        final_image = processed_img

    elif filter_option == "Sharpen":
        processed_img = img.filter(ImageFilter.SHARPEN)
        with col2:
            st.image(processed_img, caption="🔍 Sharpened Image", use_container_width=True)
        final_image = processed_img

    else:
        with col2:
            st.image(img, caption="📸 Original Image", use_container_width=True)
        final_image = img


    # 📌 Image Cropping
    st.subheader("✂️ Crop Image")
    crop_enabled = st.checkbox("Enable Cropping")
    if crop_enabled:
        # Get cropping coordinates
        left = st.slider("Left", 0, img.width - 1, 0)
        top = st.slider("Top", 0, img.height - 1, 0)
        right = st.slider("Right", left + 1, img.width, img.width)
        bottom = st.slider("Bottom", top + 1, img.height, img.height)
        
        # Crop the image
        cropped_img = final_image.crop((left, top, right, bottom))
        # Display cropped and processed images side by side
        col3, col4 = st.columns(2)
        with col3:
            st.image(cropped_img, caption="✂️ Cropped Image", use_container_width=True)
        with col4:
            st.image(final_image, caption="📸 Processed Image", use_container_width=True)
        final_image = cropped_img

    else:
        st.image(final_image, caption="📸 Processed Image", use_container_width=True)


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

# # https://growthmindset-image-app.streamlit.app/
# import streamlit as st
# import numpy as np
# from PIL import Image, ImageFilter, ImageOps
# import io

# # Streamlit App
# st.set_page_config(page_title="Image Processing App", layout="centered")

# # Custom CSS to beautify the UI
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f0f2f6;
#     }
#     .main {
#         padding: 2rem;
#     }
#     h1 {
#         color: #1e3a8a;
#         font-size: 2.5rem !important;
#     }
#     .stButton>button {
#         background-color: #1e3a8a;
#         color: white;
#         border-radius: 0.5rem;
#     }
#     .stSelectbox {
#         border-radius: 0.5rem;
#     }
#     .footer {
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100%;
#         background-color: #1e3a8a;
#         color: white;
#         text-align: center;
#         padding: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)

# st.title("🖼️ Image Processing App")
# st.write("Upload an image to apply filters, convert formats, and download.")

# # File uploader
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Open the uploaded image
#     img = Image.open(uploaded_file)
    
#     # Resize image to a smaller size
#     max_size = (400, 400)
#     img.thumbnail(max_size)
    
#     st.image(img, caption="📸 Uploaded Image (Resized)", use_container_width=True)

#     # 📌 Image Processing Filters
#     st.subheader("🎨 Apply Image Filters")
#     filter_option = st.selectbox("Choose a filter", ["Original", "Grayscale", "Blur", "Edge Detection"])

#     if filter_option == "Grayscale":
#         processed_img = ImageOps.grayscale(img)
#         st.image(processed_img, caption="🖤 Grayscale Image", use_container_width=True)
#         final_image = processed_img

#     elif filter_option == "Blur":
#         processed_img = img.filter(ImageFilter.GaussianBlur(radius=5))
#         st.image(processed_img, caption="🌫️ Blurred Image", use_container_width=True)
#         final_image = processed_img

#     elif filter_option == "Edge Detection":
#         # Convert image to grayscale first
#         gray_img = ImageOps.grayscale(img)
#         # Apply edge detection
#         edges = gray_img.filter(ImageFilter.FIND_EDGES)
#         # Invert colors to get white background
#         inverted_edges = ImageOps.invert(edges)
#         st.image(inverted_edges, caption="⚡ Edge Detection (White Background)", use_container_width=True)
#         final_image = inverted_edges

#     else:
#         st.image(img, caption="📸 Original Image", use_container_width=True)
#         final_image = img

#     # 📌 Format Conversion & Download Button
#     st.subheader("📥 Download Processed Image")
#     download_format = st.selectbox("Select download format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])

#     # Convert image to bytes
#     img_bytes_io = io.BytesIO()
#     final_image.save(img_bytes_io, format=download_format)
#     img_bytes_io.seek(0)  # Reset buffer position

#     # Download button
#     st.download_button(
#         label=f"⬇️ Download as {download_format}",
#         data=img_bytes_io,
#         file_name=f"processed_image.{download_format.lower()}",
#         mime=f"image/{download_format.lower()}",
#     )

# # Footer
# st.markdown(
#     """
#     <div class="footer">
#         Created by Nida | 
#         <a href="https://www.linkedin.com/in/nida-khurram/"  target="_blank" style="color: white;">LinkedIn</a>
#     </div>
#     """,
#     unsafe_allow_html=True
# )