"""
Synthesize accent-aware VCTK speech using the OpenVoice and MeloTTS models.

This script generates synthetic audio by cloning real VCTK speakers' voices,
using their accent metadata to guide speaker selection in the OpenVoice model.
It combines reference utterances to extract speaker embeddings, then performs
tone color conversion to match the target speaker's identity.

⚠️ Requirements:
    - The `OpenVoice` and `MeloTTS` repositories must be cloned and installed
      in the **same directory as this script** (i.e., in `preprocess/`).
    - These repos must be importable as local modules for the script to run.

Outputs:
    - Synthesized `.wav` files are saved to:
        `data/processed/synth/vctk-openvoice/raw/{speaker_id}/{transcript_id}.wav`

This script assumes prior speaker preprocessing via:
    - `vctk_resample.py` (for sample rate normalization)
    - `vctk_update_speaker_info.py` (for accent metadata)
"""

import torch
import torchaudio
import sys
from pathlib import Path
import pandas as pd
from utils import combine_wav_files, process_wav_array

# Add OpenVoice and MeloTTS repo paths to Python path for further imports
OPENVOICE_DIR = Path(__file__).resolve().parent / "OpenVoice"
MELOTTS_DIR   = Path(__file__).resolve().parent / "MeloTTS"
sys.path.insert(0, str(OPENVOICE_DIR))
sys.path.insert(0, str(MELOTTS_DIR))

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# === Configuration ===
HERE = Path(__file__).parent

# Input paths
VCTK_AUDIO_DIR    = HERE.parent / "data/processed/real/vctk-corpus-wav16/raw"
TRANSCRIPT_IDS    = HERE.parent / "data/vctk-transcript-ids.tsv"
SPEAKER_INFO_PATH = HERE.parent / "data/speaker-info-updated.csv"

# Output paths
SYNTH_DIR         = HERE.parent / "data/processed/synth/vctk-openvoice/raw"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PATH    = Path('concat.wav')  # Temp path to extract speaker embeddings

# Processing settings
TARGET_SR                  = 16000
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False  # Set to True to resynthesize

# === Load Metadata ===
speaker_info = pd.read_csv(SPEAKER_INFO_PATH)
transcripts  = pd.read_csv(TRANSCRIPT_IDS, sep='\t', dtype={"transcript_id": str})
transcripts  = dict(zip(transcripts["transcript_id"], transcripts["transcript"]))

# === Initialise Tone Color Converter for Voice Cloning (OpenVoice) ===
ckpt_converter = OPENVOICE_DIR / 'openvoice/checkpoints_v2/converter'
device         = "cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(ckpt_converter / 'config.json', device=device)
tone_color_converter.load_ckpt(ckpt_converter / 'checkpoint.pth')
tone_color_converter.watermark_model = None  # Disable watermarking

# Native sample rate from model config
native_sr = tone_color_converter.hps.data['sampling_rate']

# === Define Mapping from VCTK Speaker Accents to OpenVoice Speaker Keys ===
accent_mappings = {'British': 'EN-BR',
                   'English': 'EN-BR',
                   'Scottish': 'EN-BR',
                   'NorthernIrish': 'EN-BR',
                   'Irish': 'EN-BR',
                   'Welsh': 'EN-BR',
                   'Indian': 'EN_INDIA',  # underscore due to OpenVoice quirk
                   'Unknown': 'EN-BR',
                   'American': 'EN-US',
                   'Canadian': 'EN-US',
                   'SouthAfrican': 'EN-BR',
                   'Australian': 'EN-AU',
                   'NewZealand': 'EN-AU'
                   }

# speaker_info['ACCENTS'].unique()

#%%
# === MAIN LOOP ===
for speaker_dir in sorted(VCTK_AUDIO_DIR.iterdir()):
    if not speaker_dir.is_dir():
        continue
    
    speaker_id = speaker_dir.name
    print(f"\nProcessing speaker {speaker_id}...")
    
    audio_files = sorted(speaker_dir.glob("*.wav"))
    if not audio_files:
        continue
    
    out_dir = SYNTH_DIR / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip if already processed and not overwriting
    if not OVERWRITE and len(list(out_dir.glob("*.wav"))) >= NUM_UTTERANCES_PER_SPEAKER:
        continue
    
    # Else combine reference speaker wav files into one
    combine_wav_files(
        audio_files, REFERENCE_PATH, target_sr=TARGET_SR, silence_duration=1)
    
    # Extract target speaker embedding using the combined wav file
    target_se, _ = se_extractor.get_se(
        str(REFERENCE_PATH), tone_color_converter, vad=True)
    
    # Delete the combined wav file to preserve disk space
    REFERENCE_PATH.unlink()
    
    # Instantiate the OpenVoice model
    model = TTS(language='EN', device=device)
    
    # Determine OpenVoice speaker key from the current VCTK speaker's accent
    accent = speaker_info.loc[speaker_info['ID'] == speaker_id, 'ACCENTS'].values[0]
    speaker_key = accent_mappings.get(accent, 'EN-Default')
    
    model_speaker_ids = model.hps.data.spk2id
    model_speaker_id  = model_speaker_ids[speaker_key]
    
    # Load the speaker embedding for this OpenVoice reference speaker
    speaker_key = speaker_key.lower().replace('_', '-')
    source_se = torch.load(OPENVOICE_DIR / f'openvoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
    
    # Temporary path for model outputs before tone color conversion (cloning)
    temp_path = Path('tmp.wav')
    
    for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
        utt_id = audio_file.stem
        text   = transcripts.get(utt_id)
        
        out_path = SYNTH_DIR / speaker_id / f"{utt_id}.wav"
        if out_path.exists() and not OVERWRITE:
            continue
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            model.tts_to_file(text, model_speaker_id, temp_path, speed=1)
            
            # Run the tone color converter
            wav = tone_color_converter.convert(
                audio_src_path=temp_path, 
                src_se=source_se, 
                tgt_se=target_se,
                # output_path=out_path  # No output path returns wav array
                )
            process_wav_array(wav, native_sr, out_path, TARGET_SR)
            print(f"✓ [{speaker_id}] → {speaker_id}/{utt_id}.wav")
            
            temp_path.unlink()
        
        except Exception as e:
            print(f"✗ Error for {speaker_id}/{utt_id}: {e}")
        
print("\nDone synthesizing synthetic audio.")


# Mapped SouthAfrican to 'EN-BR' because:
# - Closest overall in terms of intonation, vowel placement, and rhythm.
# - SAE was heavily influenced by Received Pronunciation (RP) and other UK dialects during colonization.
