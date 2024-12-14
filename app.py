from flask import Flask, render_template, request
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # 前端 HTML 頁面

@app.route("/detect", methods=["POST"])
def detect():
    # 接收圖片並進行辨識
    file = request.files["file"]
    image = Image.open(file)
    processed_image = preprocess_image(np.array(image), (224, 224))
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions)
    return {"label": class_names[class_idx], "confidence": str(predictions[0][class_idx])}

if __name__ == "__main__":
    app.run(debug=True)
