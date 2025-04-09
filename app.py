import logging
from flask import Flask, request, send_file
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal
import librosa
import paho.mqtt.client as mqtt
from datetime import datetime
import json
import traceback

# ✅ Logging config
LOG_PATH = "bark_server.log"
CSV_LOG = "bark_log.csv"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
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

# 🚀 Flask app
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        raw_audio = np.frombuffer(request.data, dtype=np.uint8)

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

        print(f"[DEBUG] Upload prediction: {label} ({confidence:.2f})", flush=True)
        logging.info(f"[UPLOAD] {len(raw_audio)} bytes → {label} ({confidence:.2f})")

        with open(CSV_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{label},{confidence:.2f}\n")

        # ✅ MQTT publish block using proper JSON payload
        try:
            mqtt_payload = json.dumps({
                "label": label,
                "db": float(round(confidence * 100, 1))
            })

            print(f"[DEBUG] MQTT outgoing payload: {mqtt_payload}", flush=True)
            logging.info(f"[MQTT] Payload being sent: {mqtt_payload}")

            client = mqtt.Client(client_id="bark_server")
            client.username_pw_set("mqtt", "oxford")
            client.connect(MQTT_BROKER, 1883, 60)

            client.loop_start()
            result = client.publish(MQTT_TOPIC, mqtt_payload)
            result.wait_for_publish()
            client.loop_stop()

            print(f"[DEBUG] Published to MQTT: {mqtt_payload}", flush=True)
            logging.info(f"[MQTT] Published to topic {MQTT_TOPIC}")

            client.disconnect()

        except Exception as mqtt_error:
            print(f"[ERROR] MQTT publish failed: {mqtt_error}", flush=True)
            logging.error(f"[MQTT ERROR] {mqtt_error}")
            logging.error(traceback.format_exc())

        return f"{label} ({confidence:.2f})", 200

    except Exception as e:
        logging.error(f"[UPLOAD ERROR] {e}")
        logging.error(traceback.format_exc())
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
        logging.error(traceback.format_exc())
        return {"error": str(e)}, 500

@app.route("/log", methods=["POST"])
def log():
    try:
        message = request.data.decode("utf-8")
        logging.info(f"[ESP32] {message}")
        return '', 200
    except Exception as e:
        logging.error(f"[ESP32 LOG ERROR] {e}")
        logging.error(traceback.format_exc())
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
        logging.error(traceback.format_exc())
        return "error"

@app.route("/image")
def get_latest_image():
    path = "latest_plot.png"
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    else:
        return "No image available", 404

@app.route("/", methods=["GET"])
def root():
    return {
        "status": "bark_server running",
        "routes": ["/upload", "/predict", "/log", "/image", "/health"]
    }

@app.route("/health", methods=["GET"])
def health():
    try:
        return {"status": "ok", "model_loaded": yamnet_model is not None}, 200
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

if __name__ == "__main__":
    print("✅ Bark server running...", flush=True)
    app.run(host="0.0.0.0", port=5050)
