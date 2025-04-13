# 🐾 Bark Server — ESP32 + Flask + YAMNet + MQTT + Home Assistant

This server receives 1-second audio clips (raw 8-bit, 16kHz) from an ESP32 microphone, classifies them using TensorFlow's YAMNet model, and publishes predictions to MQTT for Home Assistant dashboards and automations.

---

## 🚀 Features
- Flask API with routes: `/upload`, `/log`, `/predict`, `/image`, `/health`
- TensorFlow + YAMNet sound classification
- MQTT output to topic: `dogmic/audio_prediction`
- CSV logging + waveform plotting
- Docker support with persistent MQTT client

---

## 📦 Project Structure

bark_server/
├── app.py               # Flask server
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── bark_log.csv         # CSV log of predictions
├── bark_server.log      # Server + MQTT debug log
├── latest_plot.png      # Spectrogram image (optional)
├── last_audio.raw       # Raw audio buffer (latest)
├── yamnet_plot.py       # Plotting util
└── test.raw             # 1s of silence for testing

---

## 📡 MQTT Format
- **Topic:** `dogmic/audio_prediction`
- **Payload Example:**
```json
{
  "label": "Domestic dogs",
  "confidence": 92.5,
  "db": 71.1
}



⸻

🧪 Local Testing

# Generate silent 1s sample (raw, 8-bit, 16kHz)
dd if=/dev/zero bs=1 count=16000 of=test.raw

# POST to upload endpoint
curl -X POST http://localhost:5050/upload \
  --data-binary @test.raw \
  -H "Content-Type: application/octet-stream"



⸻

🐳 Docker Commands

# Build container
docker build -t bark_server .

# Run with automatic restart
docker run -d \
  --name bark_server \
  --network host \
  --restart unless-stopped \
  bark_server



⸻

🧠 Home Assistant

Ensure MQTT integration is set up. Add sensors:

mqtt:
  sensor:
    - name: "Dog Bark Label"
      unique_id: dog_bark_label
      state_topic: "dogmic/audio_prediction"
      value_template: "{{ value_json.label }}"

    - name: "Dog Bark Confidence"
      unique_id: dog_bark_confidence
      state_topic: "dogmic/audio_prediction"
      value_template: "{{ value_json.confidence }}"
      unit_of_measurement: "%"
      device_class: battery

    - name: "Dog Bark Volume (dB)"
      unique_id: dog_bark_volume_db
      state_topic: "dogmic/audio_prediction"
      value_template: "{{ value_json.db }}"
      unit_of_measurement: "dB"



⸻

📊 Dashboard Ideas
	•	Markdown (Detected Sound)
	•	Gauge (dB)
	•	History Graph (Confidence)
	•	MQTT Status Tile

⸻

🧯 Troubleshooting
	•	Port conflict: sudo lsof -i :5050
	•	No MQTT logs: ensure broker IP + port (1883)
	•	Sound = Silence: check mic input or ESP signal quality

⸻

📎 Related Projects
	•	dogmic-max4466: ESP32 firmware
	•	home-assistant-config: HA dashboards + automation
	•	dogcam-esp32: Snapshot camera integration

⸻

🧠 Maintained by @MCF1981 — Part of the CottageFarm Smart Home System

MIT License
