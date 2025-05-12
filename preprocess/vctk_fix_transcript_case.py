"""
vctk_fix_transcript_case.py

This script identifies and corrects inconsistencies in capitalisation across the
VCTK corpus transcripts and lets you manually correct them via a menu.

For each group of transcripts that are identical except for casing differences, 
the user is prompted to select the preferred version. 
The script then updates the text files accordingly to ensure a consistent, canonical casing across the dataset.

Intended as a preprocessing step before transcript ID generation or training.

Usage:
    - Make sure VCTK txt files are extracted and accessible.
    - Run the script to interactively fix inconsistencies.

Dependencies:
    - tqdm
    - pathlib
    - collections
"""

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from utils import normalise_text

# Config
HERE = Path(__file__).parent
VCTK_TXT_DIR = HERE.parent / "data/corpora/VCTK-Corpus-0.92/txt"

# Group: normalised text -> list of (Path, Original Text)
lowercase_groups = defaultdict(list)

for speaker_dir in tqdm(sorted(VCTK_TXT_DIR.iterdir()), desc="Indexing speakers"):
    if not speaker_dir.is_dir():
        continue
    
    for file in sorted(speaker_dir.glob("*.txt")):
        original_text = file.read_text(encoding="utf-8").strip()
        normalised_text = normalise_text(original_text)
        lowercase_groups[normalised_text].append((file, original_text))

# Menu-driven correction
corrections = 0
for normalised, variants in lowercase_groups.items():
    unique_originals = sorted(set(original for _, original in variants))
    if len(unique_originals) <= 1:
        continue
    
    print(f"\nFound capitalisation variants for: '{normalised}'")
    for idx, variant in enumerate(unique_originals, 1):
        print(f"  [{idx}] {variant}")
    
    # Menu selection
    while True:
        try:
            choice = int(input(f"Select the correct version [1-{len(unique_originals)}]: "))
            if 1 <= choice <= len(unique_originals):
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
    
    correct_text = unique_originals[choice - 1]
    
    # Update all incorrect variants
    for path, original_text in variants:
        if original_text != correct_text:
            path.write_text(correct_text, encoding="utf-8")
            corrections += 1
            print(f"Updated {path.relative_to(VCTK_TXT_DIR)}")

print(f"\nCorrection complete. {corrections} files updated.")
