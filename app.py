import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Live KYC Verification", layout="centered")
st.title("🔐 Live Face KYC Verification (VGG-Face)")

# -----------------------------------
# Save uploaded file temporarily
# -----------------------------------
def save_file(uploaded_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(uploaded_file.read())
    temp.close()
    return temp.name

# -----------------------------------
# Blur detection
# -----------------------------------
def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# -----------------------------------
# Draw face boxes
# -----------------------------------
def draw_boxes(image_path):
    img = cv2.imread(image_path)
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="opencv",
        enforce_detection=False
    )

    for face in faces:
        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    return img, len(faces)

# -----------------------------------
# Upload ID
# -----------------------------------
id_file = st.file_uploader("Upload ID Card (PAN / Aadhar)", type=["jpg","jpeg","png"])

# -----------------------------------
# Capture Live Image
# -----------------------------------
live_image = st.camera_input("Take Live Photo")

# -----------------------------------
# Verification Logic
# -----------------------------------
if id_file and live_image:

    id_path = save_file(id_file)
    live_path = save_file(live_image)

    st.subheader("📷 Preview")

    id_img_boxed, id_count = draw_boxes(id_path)
    live_img_boxed, live_count = draw_boxes(live_path)

    st.image(id_img_boxed, caption=f"ID Image (Faces detected: {id_count})")
    st.image(live_img_boxed, caption=f"Live Image (Faces detected: {live_count})")

    if st.button("🚀 Verify Identity"):

        try:
            # Blur check
            live_img = cv2.imread(live_path)
            blur_score = check_blur(live_img)

            if blur_score < 100:
                st.warning("⚠ Live image too blurry. Please retake.")
                st.stop()

            # Face presence check
            if id_count == 0:
                st.error("❌ No face detected in ID image")
                st.stop()

            if live_count == 0:
                st.error("❌ No face detected in Live image")
                st.stop()

            if live_count > 1:
                st.error("❌ Multiple faces detected in Live image")
                st.stop()

            # DeepFace Verification
            result = DeepFace.verify(
                img1_path=id_path,
                img2_path=live_path,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )

            similarity = (1 - result["distance"]) * 100

            st.subheader("🔎 Result")

            st.write(f"Match Percentage: **{similarity:.2f}%**")

            if result["verified"]:
                st.success("✅ KYC VERIFIED")
            else:
                st.error("❌ FACE NOT MATCHING")

        except Exception as e:
            st.error(f"Verification Error: {str(e)}")

        finally:
            if os.path.exists(id_path):
                os.remove(id_path)
            if os.path.exists(live_path):
                os.remove(live_path)
