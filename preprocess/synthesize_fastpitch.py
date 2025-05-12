from pathlib import Path
import pandas as pd
from TTS.api import TTS
from utils import process_wav_array

# === Configuration ===
HERE = Path(__file__).parent

# Input paths
VCTK_AUDIO_DIR = HERE.parent / "data/processed/real/vctk-corpus-wav16/raw"
TRANSCRIPT_IDS = HERE.parent / "data/vctk-transcript-ids.tsv"

# Output folder for generated speech
SYNTH_DIR = HERE.parent / "data/processed/synth/vctk-fast-pitch/raw"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# Processing settings
MODEL_NAME = 'tts_models/en/vctk/fast_pitch'
TARGET_SR                  = 16000
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False  # Set to True to resynthesize

# === Load transcript to ID mappings ===
transcripts = pd.read_csv(TRANSCRIPT_IDS, sep="\t", dtype={"transcript_id": str})
transcripts = dict(zip(transcripts["transcript_id"], transcripts["transcript"]))

# === Initialise chosen Coqui TTS model ===
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)  # GPU bugged

# Native sample rate from model config
native_sr = tts.synthesizer.tts_config.audio["sample_rate"]

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
    
    synth_speaker_id = f"VCTK_{speaker_id}"
    if synth_speaker_id not in tts.speakers:
        continue
    
    out_dir = SYNTH_DIR / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
        utt_id = audio_file.stem
        text   = transcripts.get(utt_id)
        
        out_path = out_dir / f"{utt_id}.wav"
        if out_path.exists() and not OVERWRITE:
            continue
        
        try:
            wav = tts.tts(text=text, speaker=synth_speaker_id)
            process_wav_array(wav, native_sr, out_path, TARGET_SR)
            print(f"✓ [{speaker_id}] → {speaker_id}/{utt_id}.wav")
        
        except Exception as e:
            print(f"✗ Error for {speaker_id}/{utt_id}: {e}")

#%%
# # Alternative loop
# for synth_speaker in sorted(tts.speakers):
#     synth_speaker_id = synth_speaker.strip().split('_')[1]
    
#     real_dir = VCTK_AUDIO_DIR / synth_speaker_id
#     if not real_dir.exists():
#         continue
    
#     audio_files = sorted(real_dir.glob("*.wav"))
#     if not audio_files:
#         continue
    
#     out_dir = SYNTH_DIR / synth_speaker_id
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
#         utt_id = audio_file.stem
#         text   = transcripts.get(utt_id)
        
#         out_path = out_dir / f"{utt_id}.wav"
#         if out_path.exists() and not OVERWRITE:
#             continue
        
#         try:
#             wav = tts.tts(text=text, speaker=synth_speaker)
#             process_wav_array(wav, native_sr, out_path, TARGET_SR)
#             print(f"✓ [{synth_speaker_id}] → {synth_speaker_id}/{utt_id}.wav")
        
#         except Exception as e:
#             print(f"✗ Error for {synth_speaker_id}/{utt_id}: {e}")

# print("\nDone synthesizing synthetic audio.")
