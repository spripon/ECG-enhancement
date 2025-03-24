
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tempfile
from fpdf import FPDF

st.set_page_config(layout="wide")
st.title("Amélioration automatique d’un tracé ECG")

uploaded_file = st.file_uploader("Déposez une photo d’ECG (prise avec un smartphone)", type=["jpg", "jpeg", "png"])

def enhance_contrast_and_sharpness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def rotate_image_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text_original = pytesseract.image_to_string(gray)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    text_rotated = pytesseract.image_to_string(gray_rotated)
    keywords = ["D1", "D2", "V1", "V2", "V5", "aVL", "mm/s", "mm/mV"]
    score_original = sum(kw in text_original for kw in keywords)
    score_rotated = sum(kw in text_rotated for kw in keywords)
    return rotated if score_rotated > score_original else img

def correct_perspective(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            s = pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped
    return img

def generate_pdf(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        pil_img.save(tmp_img.name)
        path = tmp_img.name
    pdf = FPDF()
    pdf.add_page()
    pdf.image(path, x=10, y=10, w=180)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    original = img.copy()
    img = rotate_image_ocr(img)
    img = correct_perspective(img)
    img = enhance_contrast_and_sharpness(img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(original, channels="BGR", use_column_width=True)
    with col2:
        st.subheader("Image optimisée")
        st.image(img, channels="BGR", use_column_width=True)

    pdf_path = generate_pdf(img)
    with open(pdf_path, "rb") as f:
        st.download_button("Télécharger en PDF", data=f, file_name="ecg_optimise.pdf", mime="application/pdf")
