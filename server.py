from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
import mediapipe as mp

app = Flask(__name__)

# Load the pre-trained model
model = load_model("actions.keras")

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def preprocess_image(image_base64):
    # Decode the base64-encoded image
    image_data = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract keypoints using Mediapipe
    results = holistic.process(image)
    keypoints = np.zeros(1662)  # Placeholder size for keypoints
    if results.pose_landmarks:
        keypoints[:len(results.pose_landmarks.landmark)] = [
            lm.x for lm in results.pose_landmarks.landmark
        ]
    return np.expand_dims(keypoints, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_base64 = data["image"]
        input_data = preprocess_image(image_base64)

        # Make a prediction
        predictions = model.predict(input_data)[0]
        actions = ["hello", "thank_you", "i_love_you", "sorry", "no"]
        predicted_action = actions[np.argmax(predictions)]

        return jsonify({"prediction": predicted_action})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
