"""
Generate a unique ID mapping for VCTK transcript texts.

This script scans all `.txt` transcript files in the VCTK corpus directory,
assigns each **unique transcript text** a zero-padded numeric ID (e.g., "00001"),
and saves the resulting mapping as a TSV file with the following columns:

    transcript_id   transcript

The output is saved to the `TRANSCRIPT_IDS` path as `vctk-transcript-ids.tsv`.

⚠️ This script **must be run before any synthesising scripts**, as it provides a consistent
mapping from transcript text to ID, which is required to align text with synthesised audio.

➡️ It should be run **after** the following preprocessing scripts to ensure consistency:
    - `vctk_fix_transcript_case.py` (for case normalisation)
    - `vctk_fix_transcript_whitespace.py` (for whitespace cleanup)

Usage:
    - Make sure the transcript `.txt` files have been cleaned using the above scripts.
    - Run this script once to create or update the transcript ID mapping.
    - Downstream scripts will load this mapping to associate transcripts with audio.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
# from utils import normalise_text

# Paths
HERE = Path(__file__).parent
VCTK_TXT_DIR   = HERE.parent / "data/corpora/VCTK-Corpus-0.92/txt"
TRANSCRIPT_IDS = HERE.parent / "data/vctk-transcript-ids.tsv"

# Build mapping
transcript_to_id = dict()
next_id = 1

for speaker_dir in tqdm(sorted(VCTK_TXT_DIR.iterdir()), desc="Speakers"):
    for file in sorted(speaker_dir.glob("*.txt")):
        # text = normalise_text(file.read_text(encoding="utf-8"))
        text = file.read_text(encoding="utf-8").strip()
        
        # Assign an ID if this transcript is new
        if text not in transcript_to_id:
            transcript_to_id[text] = f"{next_id:05d}"  # e.g., "00001"
            next_id += 1

# Convert the transcript_to_id dict into a DataFrame
df = pd.DataFrame(list(transcript_to_id.items()),
                  columns=["transcript", "transcript_id"])

# Put transcript_id column first
df = df[["transcript_id", "transcript"]]

# Save to TSV
df.to_csv(TRANSCRIPT_IDS, sep="\t", index=False)
print(f"✅ Saved {len(df)} transcript mappings to {TRANSCRIPT_IDS}")
