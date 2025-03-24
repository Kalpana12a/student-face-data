from flask import Flask, request, jsonify
import boto3
import base64
import io
import os
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
load_dotenv()  # Load .env variables

# ✅ AWS configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "cc-student-face-data")

# ✅ Initialize AWS clients
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 🔧 Decode base64 string to OpenCV image (numpy array)
def decode_base64_image(base64_string):
    base64_data = base64_string.split(',')[-1]
    image_bytes = base64.b64decode(base64_data)
    image_np = np.array(Image.open(io.BytesIO(image_bytes)))
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 👁️ Detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"🧠 Faces found: {len(faces)}")
    return len(faces) > 0


# 📤 Save image to S3
def save_to_s3(image_np, filename):
    img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    in_mem_file = io.BytesIO()
    img.save(in_mem_file, format="JPEG")
    in_mem_file.seek(0)
    s3.upload_fileobj(in_mem_file, BUCKET_NAME, filename)
    print(f"✅ Uploaded to S3: {filename}")

# 📥 Load image from S3
def load_from_s3(filename):
    in_mem_file = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, filename, in_mem_file)
    in_mem_file.seek(0)
    image = np.array(Image.open(in_mem_file))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 🔁 Compare faces using DeepFace
def compare_faces(stored_image, live_image):
    try:
        result = DeepFace.verify(
            img1_path=stored_image,
            img2_path=live_image,
            enforce_detection=False
        )
        return {"verified": result["verified"]}
    except Exception as e:
        return {"verified": False, "error": str(e)}

@app.route("/")
def home():
    return "✅ Flask API with AWS Rekognition is running"

# ✅ Register student face to S3
@app.route('/api/register-student', methods=['POST'])
def register_face():
    data = request.get_json()
    username = data['name']
    image_data = data['image']  # base64 string

    img = decode_base64_image(image_data)
    if not detect_face(img):
        return jsonify({"status": "error", "message": "No face detected in image."}), 400

    save_to_s3(img, f"{username}.jpg")
    return jsonify({"status": "success", "message": "Face registered successfully."})


# ✅ Verify face before quiz
# ✅ Verify face before quiz using AWS Rekognition
@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    data = request.get_json()
    username = data['name']  # Same as stored filename
    image_data = data['image']
    live_image_bytes = base64.b64decode(image_data.split(",")[-1])

    try:
        response = rekognition.compare_faces(
            SourceImage={'Bytes': live_image_bytes},
            TargetImage={'S3Object': {'Bucket': BUCKET_NAME, 'Name': f"{username}.jpg"}},
            SimilarityThreshold=90
        )

        matches = response['FaceMatches']
        if matches:
            similarity = matches[0]['Similarity']
            return jsonify({"match": True, "similarity": round(similarity, 2)})
        else:
            return jsonify({"match": False, "similarity": 0})

    except Exception as e:
        print("❌ Rekognition error:", e)
        return jsonify({"match": False, "message": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
