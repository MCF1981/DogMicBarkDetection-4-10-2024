# 🐾 DogMic Audio Classification System
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-blue?logo=tensorflow&logoColor=white)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Compatible-41BDF5?logo=home-assistant&logoColor=white)
This project captures audio from an ESP32 microphone (DogMic), streams it to a Flask server running on a Raspberry Pi, analyzes the sound using Google's YAMNet model, and visualizes it as a log-mel spectrogram. Designed for bark detection, noise classification, and Home Assistant integrations.

---

## 🚀 Features

- ESP32 firmware captures 1-second, 16kHz audio clips
- Sends audio over Wi-Fi via HTTP POST to Flask server
- Flask server saves and analyzes audio with YAMNet (TensorFlow)
- Generates log-mel spectrograms with frequency/time/dB mapping
- Sends sound classification logs and alerts
- Ready for integration with Home Assistant or MQTT

<pre>
## 📦 Project Structure

```
bark_server/
├── app.py                # Flask server (upload + log endpoint)
├── yamnet_plot.py        # Plots log-mel spectrograms from .wav files
├── uploads/              # Saved audio clips (gitignored)
├── .gitignore
├── README.md             # This file
└── requirements.txt      # Python dependencies
```
</pre>


## 🔧 Setup Instructions

### 1. Clone the repo

```bash
git clone git@github.com:MCF1981/dogmic-audio-system.git
cd dogmic-audio-system

2. 🐳 Build & Run the Server in Docker
# (Re)build the Docker image
docker build -t bark-server .

# Run the server container
docker run -d \
  --name bark-server \
  --restart always \
  -p 5050:5050 \
  bark-server
3. 🔍 Check Logs
# Live log stream
docker logs bark-server --follow

# Tail the latest lines of server log (inside container)
docker exec -it bark-server tail /app/bark_server.log
4. 🔬 Test the Endpoints
# Log test message
curl -X POST http://localhost:5050/log -d "Hello from test log"

# Simulate upload (replace `test.raw` with valid raw 16kHz 8-bit audio file)
curl -X POST --data-binary @test.raw http://localhost:5050/upload

📡 ESP32 Firmware Quick Reference
ESP32 code sends:
	•	/upload: Raw 8-bit audio (16000 samples per second)
	•	/log: Text messages like status or predictions

Expect logs like this:
[ESP32] Prediction: Bark (0.82)
[UPLOAD] 16000 bytes → Bark (0.82)

🔁 Rebuilding Docker Image After Code Change
docker stop bark-server
docker rm bark-server
docker build -t bark-server .
docker run -d \
  --name bark-server \
  --restart always \
  -p 5050:5050 \
  bark-server
