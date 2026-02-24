import speech_recognition as sr
import pyttsx3
from flask import Blueprint, request, jsonify

voice_bp = Blueprint("voice", __name__)

# =========================
# TEXT TO SPEECH FUNCTION
# =========================
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # speed
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()


# =========================
# SPEECH TO TEXT FUNCTION
# =========================
def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("Recognized:", text)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech service error"


# =========================
# API ROUTES
# =========================

# Convert speech to text
@voice_bp.route("/speech-to-text", methods=["GET"])
def api_speech_to_text():
    text = speech_to_text()
    return jsonify({"text": text})


# Convert text to speech
@voice_bp.route("/text-to-speech", methods=["POST"])
def api_text_to_speech():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    speak_text(text)
    return jsonify({"message": "Speech completed"})