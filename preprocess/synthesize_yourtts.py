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
SYNTH_DIR = HERE.parent / "data/processed/synth/vctk-your-tts/raw"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# Processing settings
MODEL_NAME = 'tts_models/multilingual/multi-dataset/your_tts'
TARGET_SR                  = 16000
NUM_UTTERANCES_PER_SPEAKER = 50
OVERWRITE                  = False  # Set to True to resynthesize

# === Load Transcript to ID Mappings ===
transcripts = pd.read_csv(TRANSCRIPT_IDS, sep="\t", dtype={"transcript_id": str})
transcripts = dict(zip(transcripts["transcript_id"], transcripts["transcript"]))

# === Initialise chosen Coqui TTS model ===
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=True)

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
    
    for audio_file in audio_files[:NUM_UTTERANCES_PER_SPEAKER]:
        utt_id = audio_file.stem
        text   = transcripts.get(utt_id)
        
        out_path = SYNTH_DIR / speaker_id / f"{utt_id}.wav"
        if out_path.exists() and not OVERWRITE:
            continue
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            wav = tts.tts(text=text, speaker_wav=audio_files, language="en")
            process_wav_array(wav, native_sr, out_path, TARGET_SR)
            print(f"✓ [{speaker_id}] → {speaker_id}/{utt_id}.wav")
        
        except Exception as e:
            print(f"✗ Error for {speaker_id}/{utt_id}: {e}")
        
print("\nDone synthesizing synthetic audio.")

#%%
# from TTS.tts.utils.synthesis import synthesis
# dir(tts.synthesizer.tts_model.speaker_manager)
# embeddings = tts.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(audio_files)


# wav, alignment, _, _ = synthesis(
#     tts.synthesizer.tts_model,
#     'this is a test',
#     tts.synthesizer.tts_model.config,
#     True,
#     0,
#     use_griffin_lim=True,
#     d_vector=embeddings)



# out = tts.synthesizer.tts_model.inference(
#     text=text,
#     language="en",
#     d_vectors=embeddings
#     )
# process_coqui_wav(out["wav"], native_sr, 'test.wav', TARGET_SR)

