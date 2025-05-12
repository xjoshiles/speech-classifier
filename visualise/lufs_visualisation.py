import pyloudnorm as pyln
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# === Configuration ===
HERE = Path(__file__).parent
REAL_DIR  = HERE.parent / "data/corpora/VCTK-Corpus-0.92/wav48_silence_trimmed/p225"
SYNTH_DIR = HERE.parent / "data/processed_temp/synth"

# === Init loudness meter for synthetic files ===
synth_meter = pyln.Meter(rate=16000)

# === Collect loudness values ===
lufs_by_system = {}

for tts_dir in sorted(SYNTH_DIR.iterdir()):
    if not tts_dir.is_dir():
        continue
    system_name = tts_dir.name.split('vctk-')[1]
    wav_files = sorted((tts_dir / "raw" / "p225").glob("*.wav"))[:5]
    
    lufs_values = []
    for wav_path in wav_files:
        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # convert to mono if stereo
            loudness = synth_meter.integrated_loudness(audio)
            lufs_values.append(loudness)
        except Exception as e:
            print(f"Error reading {wav_path.name}: {e}")
    
    lufs_by_system[system_name] = lufs_values


# === Add real speech samples ===
real_wavs = sorted(REAL_DIR.glob("*mic1.flac"))[:5]  # Get first 5 alphabetically

# === Init loudness meter for real files ===
real_meter = pyln.Meter(rate=48000)

real_lufs = []
for wav_path in real_wavs:
    try:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        loudness = real_meter.integrated_loudness(audio)
        real_lufs.append(loudness)
    except Exception as e:
        print(f"Error reading real file {wav_path.name}: {e}")

lufs_by_system["real"] = real_lufs


# === Plotting ===
fig, ax = plt.subplots(figsize=(10, 6))
system_names = list(lufs_by_system.keys())
data = [lufs_by_system[sys] for sys in system_names]

ax.boxplot(data, labels=system_names, showmeans=True)
ax.set_title("LUFS Loudness Distribution across Real + Synthetic Domains")
ax.set_ylabel("Loudness (LUFS)")
ax.set_xlabel("Domain")
# ax.axhline(-23, color="red", linestyle="--", label="Target LUFS (-23)")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
