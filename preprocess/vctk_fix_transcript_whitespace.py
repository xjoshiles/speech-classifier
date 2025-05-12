"""
vctk_fix_transcript_whitespace.py

This script identifies transcripts in the VCTK corpus with consecutive whitespace 
(e.g., multiple spaces, tabs, etc.) and lets you manually correct them via a menu.

Intended as a preprocessing step before transcript ID generation or training.

Usage:
    - Navigate each problematic transcript.
    - Either accept the automatically normalised version or input your own correction.

Dependencies:
    - tqdm
    - pathlib
    - re
"""

import re
from pathlib import Path
from tqdm import tqdm
from utils import normalise_whitespace

# === Configuration ===
HERE = Path(__file__).parent
VCTK_TXT_DIR = HERE.parent / "data/corpora/VCTK-Corpus-0.92/txt"

# Collect transcripts that need fixing
problems = []
for speaker_dir in tqdm(sorted(VCTK_TXT_DIR.iterdir()), desc="Scanning transcripts"):
    if not speaker_dir.is_dir():
        continue
    
    for file in sorted(speaker_dir.glob("*.txt")):
        original = file.read_text(encoding="utf-8").strip()
        normalised = normalise_whitespace(original)
        
        if original != normalised:
            problems.append((file, original, normalised))

print(f"\nFound {len(problems)} files with whitespace issues.\n")

# Menu-driven correction
corrections = 0
for file, original, suggested in problems:
    print("=" * 60)
    print(f"File: {file.relative_to(VCTK_TXT_DIR)}\n")
    print(f"Original:  {repr(original)}\n")
    print(f"Suggested: {repr(suggested)}\n")
    
    print("Options:")
    print("1. Accept suggested fix")
    print("2. Manually edit")
    print("3. Skip")
    
    choice = input("Your choice (1/2/3): ").strip()
    
    if choice == "1":
        file.write_text(suggested, encoding="utf-8")
        print("Fixed with suggested text.\n")
        corrections += 1
        
    elif choice == "2":
        manual = input("Enter your manual correction:\n").strip()
        file.write_text(manual, encoding="utf-8")
        print("Fixed with manual text.\n")
        corrections += 1
        
    else:
        print("Skipped.\n")

print("Finished whitespace correction.")
