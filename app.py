import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal
import librosa
import paho.mqtt.publish as publish
import logging
from datetime import datetime

# Configure logging to file
logging.basicConfig(
    filename='bark_server.log',
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
from flask import Flask, request

app = Flask(__name__)

# Load YAMNet model and class names once at startup
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

# MQTT settings
MQTT_BROKER = "192.168.68.92"
MQTT_TOPIC = "dogmic/audio_prediction"
MQTT_USERNAME = "mqtt"  # 👈 replace with actual MQTT user
MQTT_PASSWORD = "oxford"  # 👈 replace with actual MQTT password

@app.route("/upload", methods=["POST"])
def upload():
    try:
        raw_audio = np.frombuffer(request.data, dtype=np.uint8)
        float_audio = (raw_audio.astype(np.float32) - 128.0) / 128.0

        # Resample to 16kHz for YAMNet
        resampled = scipy.signal.resample(float_audio, 16000)
        waveform = resampled.reshape(-1)

        # Run YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)
        scores_np = scores.numpy()
        top_class = np.argmax(np.mean(scores_np, axis=0))
        label = class_names[top_class]
        with open("bark_log.csv", "a") as f:
            f.write(f"{datetime.now().isoformat()},{label}\n")
        logging.info(f"[UPLOAD] Received {len(raw_audio)} bytes — Predicted sound: {label}") 
       # print(f"[UPLOAD] Received {len(raw_audio)} bytes — Predicted sound: {label}")

        # Publish to MQTT
        publish.single(
            MQTT_TOPIC,
            payload=label,
            hostname=MQTT_BROKER,
            auth={"username": MQTT_USERNAME, "password": MQTT_PASSWORD}
        )

        return '', 200
    except Exception as e:
        print("[ERROR] Upload failed:", e)
        return 'Upload error', 500

@app.route("/log", methods=["POST"])
def log():
    try:
        message = request.data.decode("utf-8")
        logging.info(f"[ESP32] {message}")
#       print(f"[ESP32] {message}")
        return '', 200
    except Exception as e:
        print("[ERROR] Log failed:", e)
        return 'Error', 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
