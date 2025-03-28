
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
from perspective_geometry import auto_perspective_correction

st.set_page_config(layout="wide")
st.title("Amélioration automatique des photos d’ECG")

uploaded_file = st.file_uploader("Téléchargez une photo d’ECG", type=["jpg", "jpeg", "png"])

def auto_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img[y:y+h, x:x+w]
    return img

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def upscale_image(img, scale=2):
    return cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

def save_image_as_pdf(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        pil_img.save(tmp_img.name, format="JPEG")
        tmp_img_path = tmp_img.name

    pdf = FPDF()
    pdf.add_page()
    pdf.image(tmp_img_path, x=10, y=10, w=180)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    original = image.copy()
    image = auto_perspective_correction(image)
    image = auto_crop(image)
    image = enhance_contrast(image)
    image = sharpen_image(image)
    image = upscale_image(image, scale=2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(original, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("Image optimisée")
        st.image(image, channels="BGR", use_container_width=True)

    pdf_path = save_image_as_pdf(image)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Télécharger l'image optimisée (PDF)",
            data=f,
            file_name="ecg_optimise.pdf",
            mime="application/pdf"
        )
