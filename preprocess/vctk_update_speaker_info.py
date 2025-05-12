"""
Parse and normalise VCTK speaker metadata for use in accent-aware synthesis and manifest generation.

Purpose:
    This script reads the original `speaker-info.txt` file from the VCTK corpus, which contains
    inconsistently formatted metadata (age, gender, accent, region, comments), and parses it
    into a clean, structured CSV format. It also performs known corrections, such as fixing
    malformed region labels (e.g., "English Sydney" → "Sydney").

    This processed speaker metadata is essential for downstream scripts that rely on clean and
    consistent accent and gender information.

Required Before:
    - `synthesize_openvoice.py` (for accent-aware synthetic voice generation)
    - `generate_unified_manifest.py` (for speaker-level metadata in training/inference)

Functionality:
    - Handles multi-space-delimited fields and optional comment sections in parentheses.
    - Ensures correct handling of multi-word regions and one-word accents.
    - Applies specific known corrections (e.g., REGION field standardisation).
    - Saves the structured metadata as a CSV.

Inputs:
    - VCTK `speaker-info.txt`: `data/corpora/VCTK-Corpus-0.92/speaker-info.txt`
    - (Optional) VITS speaker match file for ID alignment (currently commented out)

Output:
    - Cleaned speaker metadata saved to:
        `data/speaker-info-updated.csv`
      with columns: ["ID", "AGE", "GENDER", "ACCENTS", "REGION", "COMMENTS"]

Notes:
    - If you plan to merge VITS speaker matches (`vctk-vits-speaker-matches.csv`), a code block
      for that is included and can be uncommented when needed.
"""

from pathlib import Path
import pandas as pd
import re

# Configuration
HERE = Path(__file__).parent

SPEAKER_FILE = HERE.parent / "data/corpora/VCTK-Corpus-0.92/speaker-info.txt"
UPDATED_FILE = HERE.parent / "data/speaker-info-updated.csv"
# MATCHES_FILE = HERE.parent / "data/vctk-vits-speaker-matches.csv"


def parse_speaker_info(file_path: str) -> pd.DataFrame:
    """
    Parses the original VCTK speaker-info.txt file with inconsistent formatting.
    
    Handles:
    - Columns separated by two or more spaces
    - Multi-word REGION fields
    - COMMENTS enclosed in parentheses
    - Ensures ACCENTS is a single word
    
    Args:
        file_path (str): Path to the speaker-info.txt file.
    
    Returns:
        pd.DataFrame: A DataFrame with columns: ID, AGE, GENDER, ACCENTS, REGION, COMMENTS.
    """
    
    # Read all lines from the file
    with open(file_path, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
    
    parsed_data = []
    
    for line in raw_lines:
        # Skip headers or comment lines
        if line.strip() == "" or line.startswith(";") or line.startswith("ID"):
            continue
        
        # Extract any comment enclosed in parentheses (e.g., "(mic2 files unavailable)")
        comment_match = re.search(r"\(.*?\)", line)
        comment = comment_match.group(0) if comment_match else ""
        
        # Remove the comment from the line for cleaner parsing
        line_cleaned = re.sub(r"\(.*?\)", "", line).strip()
        
        # Split the line into parts using two or more spaces as the delimiter
        parts = re.split(r"\s{2,}", line_cleaned)
        
        # Assign known fields and combine the remaining fields into the REGION
        if len(parts) >= 4:
            ID = parts[0]
            AGE = parts[1]
            GENDER = parts[2]
            ACCENTS = parts[3]  # One-word expected
            region_parts = parts[4:]  # Remaining text is REGION
            region = " ".join(region_parts).strip()
        else:
            # If line is too short or malformed, fill missing fields with empty strings
            ID = parts[0] if len(parts) > 0 else ""
            AGE = parts[1] if len(parts) > 1 else ""
            GENDER = parts[2] if len(parts) > 2 else ""
            ACCENTS = parts[3] if len(parts) > 3 else ""
            region = ""
        
        # Append the cleaned and structured row to the result
        parsed_data.append([ID, AGE, GENDER, ACCENTS, region, comment.strip()])
    
    # Create a DataFrame with consistent column names
    df = pd.DataFrame(parsed_data, columns=["ID", "AGE", "GENDER", "ACCENTS", "REGION", "COMMENTS"])
    
    return df


# Parse original speaker-info.txt into a dataframe and fix a malformed REGION entry
speaker_df = parse_speaker_info(SPEAKER_FILE)
speaker_df['REGION'] = speaker_df['REGION'].replace('English Sydney', 'Sydney')

# Save updated speaker-info as a csv for future use
speaker_df.to_csv(UPDATED_FILE, index=False)


# # OLD CODE (merging VITS speaker ID matches)
# best_matches = pd.read_csv(MATCHES_FILE, sep=',')

# # Merge on ID
# merged_df = speaker_df.merge(
#     best_matches[["ID", "VITS_ID"]],
#     left_on="ID", right_on="ID",
#     how="left"
# )

# # Move VITS_ID right after ID
# cols = list(merged_df.columns)
# id_index = cols.index("ID")
# cols.remove("VITS_ID")
# cols.insert(id_index + 1, "VITS_ID")
# ordered_df = merged_df[cols]

# ordered_df.to_csv(UPDATED_FILE, index=False)
