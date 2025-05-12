"""
Generate unified manifests for real and synthetic VCTK audio across all feature types.

This script scans feature directories for both real and synthetic speakers (e.g., MFCCs, spectrograms, wav2vec embeddings),
and creates a unified manifest file for each feature type. Each row contains metadata about a single utterance, including:

    - File path
    - Label ("real" or "synthetic")
    - Speaker ID
    - Gender
    - Transcript ID
    - TTS system (if synthetic, else 'original')
    - Feature type

The unified manifest enables consistent downstream dataset handling, metadata access, and model training.

❗ This script must be run **after**:
    - Speaker metadata preparation (`vctk_update_speaker_info.py`)
    - Audio resampling and synthesis (real + synthetic)
    - Feature extraction (via `extract_features.py`)

❗ This script must be run **before**:
    - `generate_split_manifests.py` (which divides the data into train/val/test splits)

Inputs:
    - Processed audio feature directories in `data/processed/real/` and `data/processed/synth/`
    - Cleaned speaker info (`speaker-info-updated.csv`)

Outputs:
    - One manifest TSV per feature type, saved to `data/manifests/`:
        - `unified_raw.tsv`
        - `unified_mfcc.tsv`
        - `unified_spectrogram.tsv`
        - `unified_wav2vec.tsv`

Each manifest provides a complete, flat view of the available data, annotated with speaker and system metadata.
"""

from pathlib import Path
import pandas as pd

# Configuration
HERE = Path(__file__).parent
REAL_DIR            = HERE.parent / "data/processed/real"
SYNTH_DIR           = HERE.parent / "data/processed/synth"
SPEAKER_INFO_PATH   = HERE.parent / "data/speaker-info-updated.csv"
MANIFEST_DIR        = HERE.parent / "data/manifests"
FEATURE_TYPES       = ["raw", "mfcc", "spectrogram_2d", "wav2vec"]

# Ensure manifest directory exists
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Load speaker info
speaker_info  = pd.read_csv(SPEAKER_INFO_PATH)
gender_lookup = dict(zip(speaker_info["ID"], speaker_info["GENDER"]))

# Define synthetic TTS systems and their labels
TTS_SYSTEMS = {
    "vctk-fast-pitch": "fast-pitch",
    "vctk-openvoice": "openvoice",
    "vctk-vits": "vits",
    "vctk-xtts-v2": "xtts-v2",
    "vctk-your-tts": "your-tts"
}


def generate_manifest(feature_type: str) -> pd.DataFrame:
    manifest_rows = []
    
    print(f"\n📁 Generating manifest for feature type: {feature_type}")
    
    # === Process real data ===
    real_path = REAL_DIR / "vctk-corpus-wav16" / feature_type
    if real_path.exists():
        print(f"  ✅ Found real feature directory")
        
        for speaker_dir in real_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            gender = gender_lookup.get(speaker_id, "Unknown")
            
            for file_path in speaker_dir.glob("*"):
                if file_path.suffix not in [".wav", ".pt"]:
                    continue
                
                transcript_id = file_path.stem
                
                manifest_rows.append({
                    "path": str(file_path),
                    "label": "real",
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "transcript_id": transcript_id,
                    "tts_system": "original",
                    "feature_type": feature_type
                })
    else:
        print(f"  ⚠️ Skipping real data – directory does not exist")
    
    # === Process synthetic data ===
    for folder_name, system_label in TTS_SYSTEMS.items():
        
        synth_path = SYNTH_DIR / folder_name / feature_type
        if not synth_path.exists():
            print(f"  ⚠️ Skipping synthetic system '{system_label}' – directory missing")
            continue
        
        print(f"  ✅ Processing synthetic system '{system_label}'")
        
        for speaker_dir in synth_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            gender = gender_lookup.get(speaker_id, "Unknown")
            
            for file_path in speaker_dir.glob("*"):
                if file_path.suffix not in [".wav", ".pt"]:
                    continue
                
                transcript_id = file_path.stem
                
                manifest_rows.append({
                    "path": str(file_path),
                    "label": "synthetic",
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "transcript_id": transcript_id,
                    "tts_system": system_label,
                    "feature_type": feature_type
                })
    
    print(f"  📝 Collected {len(manifest_rows)} total samples for '{feature_type}'")
    
    return pd.DataFrame(manifest_rows)


# Generate unified manifests
output_files = {}

for feature in FEATURE_TYPES:
    manifest_df = generate_manifest(feature)
    if manifest_df.empty:
        continue
    
    out_path = MANIFEST_DIR / f"unified_{feature}.tsv"
    manifest_df.to_csv(out_path, sep='\t', index=False)
    output_files[feature] = out_path.name
