"""
Resample, normalise, and rename VCTK audio files to a standard format aligned with transcript IDs.

Purpose:
    This script standardises the VCTK corpus by applying consistent preprocessing to all speaker utterances,
    ensuring downstream models learn from speech content and not technical artifacts.
    
    The following operations are applied to each audio file:
    
    1. **Loudness Normalisation** – Adjusts perceived loudness using LUFS-based normalisation,
       ensuring volume consistency across utterances regardless of original recording levels.
       
    2. **Peak Limiting** – Applies linear peak normalisation to cap waveform amplitude at -1.0 dBFS,
       preventing clipping while preserving signal integrity.
       
    3. **Silence Trimming** – Removes leading and trailing silence based on a decibel threshold (e.g., 30 dB),
       reducing structural bias and ensuring models focus on speech content.
       
    4. **Resampling** – Converts all waveforms to a uniform sample rate of 16 kHz,
       removing variability from original recording formats and aligning with modern speech models.
    
    Processed files are renamed using transcript IDs to ensure alignment with downstream synthesis and analysis.

How It Works:
    - Loads transcript-to-ID mappings from `vctk-transcript-ids.tsv`
    - Iterates over VCTK speaker directories and reads only `*_mic1.flac` files
    - Locates the matching transcript text file for each audio file
    - Uses that transcript to assign a unique `transcript_id` as the new filename
    - Resamples the audio and saves it to `data/processed/real/vctk-corpus-wav16/raw/{speaker_id}/{transcript_id}.wav`

Required Inputs:
    - VCTK audio: `data/corpora/VCTK-Corpus-0.92/wav48_silence_trimmed/`
    - VCTK transcripts: `data/corpora/VCTK-Corpus-0.92/txt/`
    - Transcript-to-ID mapping: `data/vctk-transcript-ids.tsv`
      (generated using `vctk_generate_transcript_ids.py`)

Dependencies:
    - This script must be run **before**:
        - `vctk_synthesize_vits_for_matching.py`
        - `vctk_match_vits_speakers.py`
    - It ensures consistent sample rate and naming for real audio,
      preventing confounds in downstream embedding and speaker matching.

Output:
    - Standardised and renamed `.wav` files per speaker saved to:
        `data/processed/real/vctk-corpus-wav16/raw/{speaker_id}/{transcript_id}.wav`

Note:
    - Only audio files with available transcript `.txt` files will be processed.
    - Existing output files will be skipped unless manually deleted.
"""

from pathlib import Path
import pandas as pd
from utils import process_audio_file

# Configuration
HERE = Path(__file__).parent
VCTK_AUDIO_DIR = HERE.parent / "data/corpora/VCTK-Corpus-0.92/wav48_silence_trimmed"
TRANSCRIPT_DIR = HERE.parent / "data/corpora/VCTK-Corpus-0.92/txt"
RESAMPLE_OUTPUT_DIR = HERE.parent / "data/processed/real/vctk-corpus-wav16/raw"
TRANSCRIPT_IDS = HERE.parent / "data/vctk-transcript-ids.tsv"
TARGET_SR = 16000

# Ensure output directory exists
RESAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load transcript to ID mappings
transcripts = pd.read_csv(TRANSCRIPT_IDS, sep="\t", dtype={"transcript_id": str})
transcripts = dict(zip(transcripts["transcript"], transcripts["transcript_id"]))

#%%
# Loop through speakers and process audio
for speaker_dir in sorted(VCTK_AUDIO_DIR.iterdir()):
    if not speaker_dir.is_dir():
        continue
    
    speaker_id = speaker_dir.name  # e.g., 'p225'
    print(f"Processing speaker {speaker_id}...")
    
    # Skip speakers that do not have transcripts
    if not (TRANSCRIPT_DIR / speaker_id).exists():
        print(f"Speaker {speaker_id} has no transcript files, skipping...")
        continue
    
    # Else ensure speaker directory exists for resampled files
    speaker_resample_dir = RESAMPLE_OUTPUT_DIR / speaker_id
    speaker_resample_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files for this speaker (only mic1)
    audio_files = sorted(speaker_dir.glob("*_mic1.flac"))
    
    # Resample the files
    for audio_file in audio_files:
        filestem = audio_file.stem.split("_mic")[0]  # e.g., 'p225_001'
        
        # Get the transcript for this audio file
        transcript_path = TRANSCRIPT_DIR / speaker_id / f"{filestem}.txt"
        if not transcript_path.exists():
            print(f"WARNING: {transcript_path.name} does not exist")
            continue
        
        try:
            # Get the transcript ID for the output filename
            text = transcript_path.read_text(encoding="utf-8").strip()
            utt_id = transcripts.get(text)
            output_path = speaker_resample_dir / f"{utt_id}.wav"
            
            if output_path.exists():
                print(f"Already processed: {filestem} -> {utt_id}")
                continue
            
            process_audio_file(audio_file, output_path, target_sr=TARGET_SR)
            print(f"Standardised and renamed: {filestem} -> {utt_id}")
        
        except Exception as e:
            print(f"Failed for {filestem}: {e}")
