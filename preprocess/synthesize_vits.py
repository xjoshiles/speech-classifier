from pathlib import Path
import pandas as pd
from TTS.api import TTS
from utils import process_wav_array

# === Configuration ===
HERE = Path(__file__).parent

# Input paths
VCTK_AUDIO_DIR = HERE.parent / "data/processed/real/vctk-corpus-wav16/raw"
TRANSCRIPT_IDS = HERE.parent / "data/vctk-transcript-ids.tsv"
MATCHES_FILE   = HERE.parent / "data/vctk-vits-speaker-matches.csv"

# Output folder for generated speech
SYNTH_DIR = HERE.parent / "data/processed/synth/vctk-vits/raw"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# Processing settings
MODEL_NAME = 'tts_models/en/vctk/vits'
TARGET_SR                  = 16000
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False  # Set to True to resynthesize

# === Load transcript to ID mappings ===
transcripts = pd.read_csv(TRANSCRIPT_IDS, sep="\t", dtype={"transcript_id": str})
transcripts = dict(zip(transcripts["transcript_id"], transcripts["transcript"]))

# Load speaker metadata and exclude incorrect matches
speaker_matches   = pd.read_csv(MATCHES_FILE).dropna(subset=["VITS_ID"])
tts_id_to_real_id = dict(zip(speaker_matches["VITS_ID"], speaker_matches["ID"]))

# === Initialise chosen Coqui TTS model ===
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=True)

# Native sample rate from model config
native_sr = tts.synthesizer.tts_config.audio["sample_rate"]

#%%
# === MAIN LOOP ===
for synth_speaker in sorted(tts.speakers):
    synth_speaker_id = synth_speaker.strip()
    
    real_speaker_id = tts_id_to_real_id.get(synth_speaker_id, False)
    if not real_speaker_id:
        continue
    print(f"\nProcessing speaker {real_speaker_id}...")
    
    real_dir = VCTK_AUDIO_DIR / real_speaker_id
    if not real_dir.exists():
        continue
    
    audio_files = sorted(real_dir.glob("*.wav"))
    if not audio_files:
        continue
    
    out_dir = SYNTH_DIR / real_speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
        utt_id = audio_file.stem
        text   = transcripts.get(utt_id)
        
        out_path = out_dir / f"{utt_id}.wav"
        if out_path.exists() and not OVERWRITE:
            continue
        
        try:
            wav = tts.tts(text=text, speaker=synth_speaker)
            process_wav_array(wav, native_sr, out_path, TARGET_SR)
            print(f"✓ [VITS {synth_speaker_id}] → {real_speaker_id}/{utt_id}.wav")
        
        except Exception as e:
            print(f"✗ Error for {real_speaker_id}/{utt_id}: {e}")

print("\nDone synthesizing synthetic audio.")
