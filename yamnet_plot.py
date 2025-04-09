import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import os
import soundfile as sf

def convert_raw_to_wav(raw_path, wav_path, sample_rate=16000):
    print(f"🔄 Converting {raw_path} → {wav_path} ...")
    raw = np.fromfile(raw_path, dtype=np.uint8)
    audio = (raw.astype(np.float32) - 128.0) / 128.0  # Normalize to -1.0 to +1.0
    sf.write(wav_path, audio, samplerate=sample_rate)
    print("✅ Conversion complete!")

def plot_log_mel(filepath):
    print(f"📂 Loading {filepath}...")
    y, sr = librosa.load(filepath, sr=16000)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-Mel Spectrogram\n{os.path.basename(filepath)}')
    plt.tight_layout()
    plt.show()
    plt.savefig("latest_plot.png", dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage:")
        print("   python yamnet_plot.py file.wav")
        print("   python yamnet_plot.py file.raw (auto-converts to .wav)")
        sys.exit(1)

    path = sys.argv[1]

    if path.endswith(".raw"):
        wav_path = path.replace(".raw", ".wav")
        convert_raw_to_wav(path, wav_path)
        plot_log_mel(wav_path)
    else:
        plot_log_mel(path)
