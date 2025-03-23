
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import pytesseract

st.set_page_config(layout="wide")
st.title("Amélioration automatique des photos d’ECG")

uploaded_file = st.file_uploader("Téléchargez une photo d’ECG", type=["jpg", "jpeg", "png"])

def detect_orientation_and_rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh)
    keywords = ["D1", "D2", "aVL", "V2", "V5", "25 mm/s", "10 mm/mV"]
    matches = [kw for kw in keywords if kw in text]

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh_rotated = cv2.threshold(gray_rotated, 180, 255, cv2.THRESH_BINARY_INV)
    text_rotated = pytesseract.image_to_string(thresh_rotated)
    matches_rotated = [kw for kw in keywords if kw in text_rotated]

    if len(matches_rotated) > len(matches):
        return rotated
    return img

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def apply_perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def perspective_correction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return apply_perspective_transform(img, pts)
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
    image = detect_orientation_and_rotate(image)
    image = perspective_correction(image)
    image = enhance_contrast(image)
    image = sharpen_image(image)
    image = upscale_image(image, scale=2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(original, channels="BGR", use_column_width=True)
    with col2:
        st.subheader("Image optimisée")
        st.image(image, channels="BGR", use_column_width=True)

    pdf_path = save_image_as_pdf(image)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Télécharger l'image optimisée (PDF)",
            data=f,
            file_name="ecg_optimise.pdf",
            mime="application/pdf"
        )
