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
