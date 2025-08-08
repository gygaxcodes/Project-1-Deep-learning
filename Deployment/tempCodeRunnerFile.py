import os
from flask import Flask, request, jsonify, render_template

from pyngrok import ngrok
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# Load the model
model = load_model("model/model_3.keras")

class_names = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]

# Flask setup
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        prediction = model.predict(image)[0]
        top_index = int(np.argmax(prediction))
        result = {
            "prediction": class_names[top_index],
            "confidence": float(prediction[top_index]),
            "all_probs": {
                class_names[i]: float(prediction[i]) for i in range(len(class_names))
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Start ngrok + Flask
ngrok.set_auth_token("3106RQFbSkY3z9PwTLYHKr0q9A3_6iK8CffmCPGvcAuNzeT19")
public_url = ngrok.connect(5000)
print(f"🚀 App is live at: {public_url}")
app.run(port=5000)
