import logging
from flask import Flask, request
from flask import send_file
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal
import librosa
import paho.mqtt.publish as publish
from datetime import datetime

# ✅ Logging config
LOG_PATH = "bark_server.log"
CSV_LOG = "bark_log.csv"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)

logging.info("✅ Bark server started.")

# 🧠 Load YAMNet and labels
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

# 📡 MQTT config
MQTT_BROKER = "192.168.68.92"
MQTT_TOPIC = "dogmic/audio_prediction"
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "oxford"

# 🚀 Flask app
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        raw_audio = np.frombuffer(request.data, dtype=np.uint8)

        # Save latest audio for plotting
        with open("last_audio.raw", "wb") as f:
            f.write(request.data)

        float_audio = (raw_audio.astype(np.float32) - 128.0) / 128.0
        resampled = scipy.signal.resample(float_audio, 16000)
        waveform = resampled.reshape(-1)

        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_class = np.argmax(mean_scores)
        label = class_names[top_class]
        confidence = mean_scores[top_class]

        logging.info(f"[UPLOAD] {len(raw_audio)} bytes → {label} ({confidence:.2f})")

        with open(CSV_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{label},{confidence:.2f}\n")

        publish.single(
            MQTT_TOPIC,
            payload=label,
            hostname=MQTT_BROKER,
            auth={"username": MQTT_USERNAME, "password": MQTT_PASSWORD}
        )

        return f"{label} ({confidence:.2f})", 200

    except Exception as e:
        logging.error(f"[UPLOAD ERROR] {e}")
        return 'Upload error', 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        audio_chunk = request.data
        label = classify_audio(audio_chunk)

        if label.lower() == "bark":
            logging.info("🐶 Bark detected.")
        elif label.lower() == "silence":
            logging.info("🔇 Silence detected.")
        else:
            logging.info(f"🔍 Detected: {label}")

        return {"prediction": label}, 200

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return {"error": str(e)}, 500

@app.route("/log", methods=["POST"])
def log():
    try:
        message = request.data.decode("utf-8")
        logging.info(f"[ESP32] {message}")
        return '', 200
    except Exception as e:
        logging.error(f"[ESP32 LOG ERROR] {e}")
        return 'Error', 500

def classify_audio(audio_bytes):
    try:
        float_audio = (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        waveform = scipy.signal.resample(float_audio, 16000)
        waveform = waveform.reshape(-1)

        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_class = np.argmax(mean_scores)
        label = class_names[top_class]
        return label

    except Exception as e:
        logging.error(f"[CLASSIFY ERROR] {e}")
        return "error"

@app.route("/image")
def get_latest_image():
    path = "latest_plot.png"  # Adjusted to match local filename
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    else:
        return "No image available", 404

@app.route("/", methods=["GET"])
def root():
    return {
        "status": "bark-server running",
        "routes": ["/upload", "/predict", "/log", "/image", "/health"]
    }

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model_loaded": yamnet_model is not None}, 200

if __name__ == "__main__":
    print("✅ Bark server running...")
    app.run(host="0.0.0.0", port=5050)
