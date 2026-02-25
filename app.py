from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
import pytesseract
from voice import voice_bp
import os
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# =========================
# APP CONFIG
# =========================

app = Flask(__name__)
CORS(app)

app.register_blueprint(voice_bp)

# =========================
# LOAD SIGN MODEL
# =========================

try:
    model = load_model("sign_model.h5")
    print("✅ Sign model loaded successfully.")
except Exception as e:
    model = None
    print("⚠ Model load error:", e)

labels = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]

# =========================
# PAGE ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/voice")
def voice():
    return render_template("voice.html")

@app.route("/booking")
def booking():
    return render_template("booking.html")

@app.route("/location")
def location():
    return render_template("location.html")

@app.route("/ocr")
def ocr():
    return render_template("ocr.html")

@app.route("/sign")
def sign():
    return render_template("sign.html")

# =========================
# SIGN PREDICTION API
# =========================

@app.route("/predict_sign", methods=["POST"])
def predict_sign():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json["image"]

        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Improved skin range (better for Indian skin tones)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        # Extract hand region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return jsonify({"prediction": "-", "confidence": 0})

        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped = img[y:y+h, x:x+w]

        # White background canvas
        white_bg = np.ones((224, 224, 3), dtype=np.uint8) * 255

        # Maintain aspect ratio
        scale = 200 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(cropped, (new_w, new_h))

        start_x = (224 - new_w) // 2
        start_y = (224 - new_h) // 2

        white_bg[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        input_img = white_bg / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        prediction = model.predict(input_img)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label = labels[class_index]

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# OCR API
# =========================

@app.route("/predict_ocr", methods=["POST"])
def predict_ocr():

    try:
        data = request.json["image"]

        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        cleaned_text = ''.join(
            char for char in text
            if char.isalnum() or char in "@#$%&*()_-+=!?.,:; "
        )

        return jsonify({"text": cleaned_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# BOOKING STORAGE (DEMO)
# =========================

bookings = []

@app.route("/save_booking", methods=["POST"])
def save_booking():

    try:
        data = request.json

        booking_id = f"BK{len(bookings)+1}"

        booking_data = {
            "id": booking_id,
            "name": data.get("name"),
            "service": data.get("service"),
            "date": data.get("date"),
            "time": data.get("time"),
            "location": data.get("location"),
            "status": "Confirmed"
        }

        bookings.append(booking_data)

        return jsonify({
            "message": "Booking Saved",
            "booking": booking_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_bookings", methods=["GET"])
def get_bookings():
    return jsonify(bookings)


@app.route("/update_booking", methods=["POST"])
def update_booking():

    try:
        data = request.json
        booking_id = data.get("id")
        new_status = data.get("status")

        for booking in bookings:
            if booking["id"] == booking_id:
                booking["status"] = new_status
                break

        return jsonify({"message": "Booking Updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
