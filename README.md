# 🐾 Bark Server — Flask + TensorFlow YAMNet Audio Classifier

This is the Dockerized Flask server that accepts 1-second audio clips (raw 8-bit, 16kHz), classifies them using YAMNet (TensorFlow), and publishes predictions to MQTT for Home Assistant automations.

## 🚀 Features

- Flask server with endpoints: `/upload`, `/log`, `/predict`, `/image`, `/health`
- Classifies sounds via [YAMNet](https://tfhub.dev/google/yamnet/1) from TensorFlow Hub
- Publishes prediction to `dogmic/audio_prediction` via MQTT
- Generates `latest_plot.png` and `bark_log.csv` for logging
- Fully containerized via Docker

## 📦 Project Structure
bark_server/
├── app.py                # Flask app
├── requirements.txt      # Python deps
├── bark_log.csv          # CSV of all predictions
├── bark_server.log       # Server logs
├── yamnet_plot.py        # Spectrogram plotting
├── last_audio.raw/.wav   # Most recent raw + converted audio
├── latest_plot.png       # Most recent spectrogram
└── Dockerfile
---

## 🐳 Docker Usage

# Build the image
docker build -t bark-server .

# Run the container
docker run -d \
  --name bark-server \
  -p 5050:5050 \
  --restart always \
  bark-server

## 🔬 API Endpoints

| Method | Endpoint   | Description                         |
|--------|------------|-------------------------------------|
| POST   | `/upload`  | Raw 8-bit 1s audio, returns label   |
| POST   | `/predict` | Prediction from audio chunk         |
| POST   | `/log`     | Logs from ESP32                     |
| GET    | `/image`   | Returns `latest_plot.png`           |
| GET    | `/health`  | Returns model + server status       |

## 🧪 Test It Locally
# Generate silent test audio (1s @ 16kHz)
dd if=/dev/zero bs=1 count=16000 of=test.raw

# Send it to /upload
curl -X POST http://localhost:5050/upload \
  --data-binary @test.raw \
  -H "Content-Type: application/octet-stream"

⸻

🏠 Home Assistant Integration
	•	MQTT Topic: dogmic/audio_prediction
	•	Broker: 192.168.68.92 (HA IP)
	•	Example HA sensors:
	•	binary_sensor.bark_detected
	•	sensor.bark_detected_status
	•	sensor.dog_mic_volume_db


⸻

📊 Dashboard Cards

See dogmic-system/dashboards/dogmic-dashboard.yaml
	•	Mic Volume (Gauge)
	•	Prediction (Markdown)
	•	Snapshot preview
	•	Tunnel health status

⸻

🧯 Recovery Notes
	•	If HA tunnel breaks, restart Cloudflared add-on
	•	Rebuild Flask container:
docker stop bark-server
docker rm bark-server
docker build -t bark-server .
docker run -d --name bark-server -p 5050:5050 --restart always bark-server

🧠 Maintained by @MCF1981

Part of the CottageFarm Smart Home System
MIT License
---

### ✅ Next Steps:

```bash
cd ~/bark_server
nano README.md      # paste the above into the file
Then:
git add README.md
git commit -m "📖 Clean up Bark Server README with Docker, HA, MQTT integration"
git push
