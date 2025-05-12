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
SYNTH_DIR = HERE.parent / "data/processed/synth/vctk-xtts-v2/raw"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# Processing settings
MODEL_NAME = 'tts_models/multilingual/multi-dataset/xtts_v2'
TARGET_SR                  = 16000
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False  # Set to True to resynthesize

# === Load Transcript to ID Mappings ===
transcripts = pd.read_csv(TRANSCRIPT_IDS, sep="\t", dtype={"transcript_id": str})
transcripts = dict(zip(transcripts["transcript_id"], transcripts["transcript"]))

# === Initialise chosen Coqui TTS model ===
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)  # (CPU faster)

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
    
    out_dir = SYNTH_DIR / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip if already processed and not overwriting
    if not OVERWRITE and len(list(out_dir.glob("*.wav"))) >= NUM_UTTERANCES_PER_SPEAKER:
        continue
    
    # Else precompute speaker embedding
    gpt_latent, speaker_emb = tts.synthesizer.tts_model.get_conditioning_latents(
        audio_path=audio_files,
        load_sr=TARGET_SR  # the sample rate of the audio file(s)
        )
    
    for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
        utt_id = audio_file.stem
        text   = transcripts.get(utt_id)
        
        out_path = SYNTH_DIR / speaker_id / f"{utt_id}.wav"
        if out_path.exists() and not OVERWRITE:
            continue
        
        try:
            out = tts.synthesizer.tts_model.inference(
                text=text,
                language="en",
                gpt_cond_latent=gpt_latent,
                speaker_embedding=speaker_emb
                )
            process_wav_array(out["wav"], native_sr, out_path, TARGET_SR)
            print(f"✓ [{speaker_id}] → {speaker_id}/{utt_id}.wav")
        
        except Exception as e:
            print(f"✗ Error for {speaker_id}/{utt_id}: {e}")

print("\nDone synthesizing synthetic audio.")


# CPU fastest, multithreading impossible due to system stall 
