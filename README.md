.

🔐 AI-Based Digital KYC Face Verification System

An AI-powered Digital KYC (Know Your Customer) verification system that performs real-time facial identity validation by comparing an uploaded ID card image with a live camera capture.

Built using DeepFace (VGG-Face model) for deep learning-based face verification and Streamlit for an interactive web interface.

🚀 Features

📄 Upload ID card image (PAN / Aadhaar format / Passport Size photo)

📷 Capture live image using camera

🧠 Deep learning-based facial verification (VGG-Face)

🔍 Face detection with bounding box visualization

📊 Match percentage calculation

⚠ Blur detection for image quality validation

🚫 Multiple face prevention logic

🗑 Automatic temporary file cleanup

🧠 How It Works

User uploads an ID card image.

User captures a live photo.

System performs:

Face detection using OpenCV

Blur detection using Laplacian variance

Face count validation

DeepFace generates facial embeddings.

Distance-based similarity comparison is performed.

Verification result is displayed.

🛠 Tech Stack

Python

Streamlit

OpenCV

DeepFace (VGG-Face)

NumPy

Pillow
