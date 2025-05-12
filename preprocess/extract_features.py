"""
Extract audio features (MFCC, spectrogram, and Wav2Vec) for all real and synthetic VCTK speakers.

This script processes resampled real speech and synthesized speech data to compute standardized
audio features for use in training and evaluation pipelines. It supports multiple feature types,
ensures consistent frame length per sample, and saves `.pt` tensors for each utterance.

❗ This script must be run **after**:
    - Real audio has been resampled via `vctk_resample.py`
    - Synthetic audio has been generated (e.g. `synthesize_vits.py`)

❗ This script must be run **before**:
    - Generating dataset manifests (e.g., via `generate_unified_manifest.py`)
    - Initializing dataloaders that consume extracted feature tensors

Supported feature types:
    - "mfcc"
    - "spectrogram_2d"
    - "wav2vec" (using Hugging Face Wav2Vec2Model)

Outputs:
    - For each utterance, a feature tensor saved as:
        {dataset_dir}/{feature_type}/{speaker_id}/{utterance_id}.pt

Configuration:
    - Max frames: feature tensors are padded or truncated to `MAX_FRAMES`
    - Re-extraction is skipped unless `OVERWRITE = True`

This script must be rerun if any upstream audio or transcript processing changes.
"""

import torch
from pathlib import Path
from tqdm import tqdm

from features import get_extractor

# === Configuration ===
HERE = Path(__file__).parent

# Input paths
ROOT_REAL_DIR  = HERE.parent / "data/processed/real"
ROOT_SYNTH_DIR = HERE.parent / "data/processed/synth"

# Processing settings
FEATURE_TYPES              = ["mfcc", "spectrogram_2d", "wav2vec"]
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False
MAX_FRAMES                 = 320

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting feature extraction on {DEVICE.upper()}")
print(f"Feature types: {', '.join([ft.upper() for ft in FEATURE_TYPES])}")

# === Loop through each feature type ===
for feature_type in FEATURE_TYPES:
    print(f"\n=== Extracting features: {feature_type.upper()} ===")
    extractor = get_extractor(feature_type, device=DEVICE)
    
    # === Loop through real and synth directories ===
    for mode, root_dir in zip(["real", "synth"], [ROOT_REAL_DIR, ROOT_SYNTH_DIR]):
        print(f"🔎 Processing {mode} data from: {root_dir}")
        
        for dataset_dir in tqdm(sorted(root_dir.iterdir()), desc=f"Datasets ({mode})"):
            if not dataset_dir.is_dir():
                continue
            
            raw_dir = dataset_dir / "raw"
            if not raw_dir.exists():
                print(f"⚠️  Warning: {raw_dir} does not exist, skipping.")
                continue
            
            for speaker_dir in tqdm(sorted(raw_dir.iterdir()), desc=f"Speakers ({dataset_dir.name})", leave=False):
                if not speaker_dir.is_dir():
                    continue
                speaker_id = speaker_dir.name
                
                wav_files = list(sorted(speaker_dir.glob("*.wav")))[:NUM_UTTERANCES_PER_SPEAKER]
                
                for wav_path in tqdm(wav_files, desc=f"Utterances ({speaker_id})", leave=False):
                    trans_id = wav_path.stem
                    
                    # Define output feature path
                    out_dir = dataset_dir / feature_type / speaker_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    feature_path = out_dir / f"{trans_id}.pt"
                    
                    # Extract and save feature if needed
                    if OVERWRITE or not feature_path.exists():
                        extractor.extract_and_save(wav_path, feature_path, max_frames=MAX_FRAMES)
