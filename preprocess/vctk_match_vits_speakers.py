"""
Match real VCTK speaker IDs to synthetic speaker IDs generated using the Coqui VITS model.

Purpose:
    This script compares speaker embeddings between real VCTK utterances and their synthetic
    counterparts (produced via `vctk_synthesize_vits_for_matching.py`) in order to determine
    which synthetic VITS speaker best matches each real speaker.

    It addresses a known issue in the Coqui VITS model where real and synthetic speaker identities
    were misaligned during training:
        https://github.com/coqui-ai/TTS/issues/2258

    The final output is a cleaned mapping of real speaker IDs to VITS speaker IDs, with low-confidence
    matches excluded, ready for use in downstream synthesis pipelines.

Required Preprocessing:
    - Run `vctk_fix_transcript_case.py` and `vctk_fix_transcript_whitespace.py`
    - Generate transcript IDs via `generate_vctk_transcript_ids.py`
    - Resample all VCTK `.wav` files to a consistent sample rate using `vctk_resample.py`
    - Generate synthetic speech for all speakers using `vctk_synthesize_vits_for_matching.py`

How It Works:
    - Embeds real and synthetic utterances using `resemblyzer`.
    - Computes pairwise cosine similarity between real and synthetic speakers using both:
        1. Per-utterance similarity (average over aligned utterance pairs)
        2. Mean speaker embedding similarity
    - Reports unmatched and duplicate VITS matches, and selects the more consistent mapping.
    - Optionally removes low-confidence matches (e.g. weakest match in duplicate cases).

Output:
    - A cleaned speaker match file saved to:
        `data/vctk-vits-speaker-matches.csv`
      with columns: `["ID", "VITS_ID"]`

Next Steps:
    - Use this mapping to drive speaker-aware synthesis and evaluation pipelines.
    - Review any NaN entries manually if applicable (indicating uncertain matches).

This script is essential for establishing a reliable mapping between real and synthetic VCTK speakers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from resemblyzer import VoiceEncoder, preprocess_wav
from collections import Counter
from tqdm import tqdm

# === Configuration ===
HERE = Path(__file__).parent
REAL_AUDIO_DIR = HERE.parent / "data/processed/real/vctk-corpus-wav16/raw"
SYNTH_DIR      = HERE.parent / "data/temp/synth/vctk-vits/raw"
MATCHES_FILE   = HERE.parent / "data/vctk-vits-speaker-matches.csv"
NUM_UTTERANCES_PER_SPEAKER = 20

encoder = VoiceEncoder()


def embed_utterances(audio_dir: Path,
                     max_utts: int = None,
                     desc: str = "Processing") -> dict:
    """
    Generate speaker-wise embeddings for utterances in a directory.

    This function processes a directory where each subdirectory represents a speaker,
    containing `.wav` audio files of that speaker’s utterances. Each file is preprocessed
    and passed through the voice encoder to generate embeddings.

    Parameters:
        audio_dir (Path): Path to the parent directory containing speaker subdirectories.
        max_utts (int, optional): Maximum number of utterances to process per speaker.
                                  If None, all `.wav` files are used.
        desc (str): Description to show in the tqdm progress bar.

    Returns:
        dict: A nested dictionary mapping speaker IDs to utterance embeddings:
              {
                  "speaker_id": {
                      "utterance_id": embedding_vector,
                      ...
                  },
                  ...
              }
    """
    embeddings = {}
    
    # Iterate over each speaker subdirectory
    for speaker_dir in tqdm(sorted(audio_dir.iterdir()), desc=desc):
        if not speaker_dir.is_dir():
            continue  # Skip non-directory files
        
        speaker_id = speaker_dir.name
        files = sorted(speaker_dir.glob("*.wav"))
        
        # Optionally limit the number of utterances
        if max_utts is not None:
            files = files[:max_utts]
        
        utt_dict = {}
        
        # Process each utterance file
        for f in files:
            try:
                wav = preprocess_wav(f)  # Convert to mono, normalise, etc.
                utt_dict[f.stem] = encoder.embed_utterance(wav)
            except Exception as e:
                print(f"[ERROR] Failed to process {f}: {e}")
        
        # Store embeddings if any valid utterances were found
        if utt_dict:
            embeddings[speaker_id] = utt_dict
    
    return embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_similarity(real_embs, synth_embs, use_mean=False):
    """
    Compute cosine similarity between real and synthetic speaker embeddings.

    Parameters:
        real_embs (dict): Dictionary mapping real speaker IDs to dicts of utterance embeddings.
                          Format: {speaker_id: {utterance_id: embedding_vector}}
        synth_embs (dict): Dictionary mapping synthetic speaker IDs to dicts of utterance embeddings.
        use_mean (bool): If True, compute similarity between mean embeddings per speaker.
                         If False, compute mean similarity across matching utterance pairs.

    Returns:
        pd.DataFrame: Columns = ["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
    """
    results = []

    # Iterate through all combinations of real and synthetic speakers
    for real_id, real_utts in real_embs.items():
        for synth_id, synth_utts in synth_embs.items():
            
            # Find shared utterance IDs between real and synthetic sets
            common_utts = sorted(set(real_utts) & set(synth_utts))
            if not common_utts:
                continue  # Skip if no matching utterances
            
            if use_mean:
                # Compare based on average embedding over shared utterances
                mean_real = np.mean([real_utts[utt] for utt in common_utts], axis=0)
                mean_synth = np.mean([synth_utts[utt] for utt in common_utts], axis=0)
                sim = cosine_similarity(mean_real, mean_synth)
                results.append((real_id, synth_id, sim, len(common_utts)))
            else:
                # Compare each pair of shared utterances individually, then average
                sims = [
                    cosine_similarity(real_utts[utt], synth_utts[utt])
                    for utt in common_utts
                ]
                mean_sim = np.mean(sims)
                results.append((real_id, synth_id, mean_sim, len(common_utts)))
    
    # Return as a DataFrame
    return pd.DataFrame(results, columns=["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"])


def analyse_results(df: pd.DataFrame):
    """
    Analyse a similarity DataFrame to extract best matches, unmatched synthetic IDs, and duplicates.
    
    Parameters:
        df (pd.DataFrame): Full similarity DataFrame with columns:
            ["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
    
    Returns:
        Tuple[pd.DataFrame, List[str], pd.Series]:
            - best_matches (DataFrame): Best matching synthetic speaker for each real speaker.
            - unmatched_synthetic_ids (List[str]): Synthetic speaker IDs not matched as any best match.
            - duplicate_matches (Series): Synthetic speaker IDs matched to more than one real speaker.
    """
    df = df.copy()
    df.sort_values(by=["ID", "Mean_Cosine_Similarity"], ascending=[True, False], inplace=True)
    
    # Extract the best match for each real speaker
    best_matches = df.groupby("ID").first().reset_index()
    
    # Identify synthetic speakers that were never a top match
    all_synth_ids = set(df["VITS_ID"])
    matched_synth_ids = set(best_matches["VITS_ID"])
    unmatched_synthetic_ids = sorted(all_synth_ids - matched_synth_ids)
    
    # Find synthetic speakers matched to more than one real speaker
    duplicate_matches = best_matches["VITS_ID"].value_counts()
    duplicate_matches = duplicate_matches[duplicate_matches > 1]
    
    return best_matches, unmatched_synthetic_ids, duplicate_matches


def analyse_reverse_matches(df, unmatched_synthetic_ids, best_matches):
    """
    For each unmatched synthetic speaker (VITS_ID), find the real speaker (ID) they are most similar to,
    and compare that real speaker’s actual best-matched VITS_ID to identify conflicts.
    
    Parameters:
        df (pd.DataFrame): Full similarity DataFrame with columns:
            ["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
        unmatched_synthetic_ids (List[str]): VITS_IDs that were not selected as best match.
        best_matches (pd.DataFrame): DataFrame of best matches per real speaker.
            Must include columns: ["ID", "VITS_ID"]
    
    Returns:
        pd.DataFrame: Columns: ["ID", "VITS_ID", "Reverse_Similarity", "Best_VITS_ID", "Conflict"]
    """
    df = df.copy()
    
    # Filter to unmatched synths and find their best matching real speaker (highest similarity)
    reverse_best_matches = (
        df[df["VITS_ID"].isin(unmatched_synthetic_ids)]
        .sort_values(by=["VITS_ID", "Mean_Cosine_Similarity"], ascending=[True, False])
        .groupby("VITS_ID")
        .first()
        .reset_index()
    )
    
    # Merge to get the real speaker's actual best VITS_ID
    merged = reverse_best_matches.merge(
        best_matches[["ID", "VITS_ID"]],
        on="ID",
        how="left",
        suffixes=("", "_Best")
    )
    
    # Rename columns and flag conflicts
    merged = merged.rename(columns={
        "VITS_ID": "VITS_ID",               # The unmatched VITS speaker
        "VITS_ID_Best": "Best_VITS_ID",     # Who the real speaker actually picked
        "Mean_Cosine_Similarity": "Reverse_Similarity"
    })
    merged["Conflict"] = merged["VITS_ID"] != merged["Best_VITS_ID"]
    
    return merged[["ID", "VITS_ID", "Reverse_Similarity", "Best_VITS_ID", "Conflict"]]


def compare_duplicate_matches(df, best_matches, duplicate_matches):
    """
    For synthetic speakers (VITS_IDs) that are matched to multiple real speakers (IDs),
    list all real speakers who selected them, along with similarity scores for inspection.
    
    Parameters:
        df (pd.DataFrame): Full similarity DataFrame with columns:
            ["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
        best_matches (pd.DataFrame): Best match per real speaker.
            Must contain columns ["ID", "VITS_ID"].
        duplicate_matches (pd.Series): VITS_IDs that appear as best match for more than one real speaker.

    Returns:
        pd.DataFrame: A DataFrame showing all real speakers that matched to duplicate synths,
                      along with their similarity scores.
                      Columns: ["VITS_ID", "ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
    """
    df = df.copy()
    
    # Get just the VITS_IDs that were matched to more than one real speaker
    duplicate_synth_ids = duplicate_matches.index.tolist()
    
    # Filter best matches to only the duplicates
    duplicates_df = best_matches[best_matches["VITS_ID"].isin(duplicate_synth_ids)]
    
    # Join back with the full results to get similarity scores
    merged = pd.merge(
        duplicates_df[["ID", "VITS_ID"]],
        df,
        on=["ID", "VITS_ID"],
        how="left"
    )
    
    # Sort for readability
    merged = merged.sort_values(by=["VITS_ID", "Mean_Cosine_Similarity"], ascending=[True, False])
    
    return merged[["ID", "VITS_ID", "Mean_Cosine_Similarity", "Utterances_Compared"]]


def get_lowest_score_ids(duplicates_df):
    """
    Identify the real speakers (ID) with the lowest similarity score to each duplicate synthetic speaker (VITS_ID).
    
    Parameters:
        duplicates_df (pd.DataFrame): DataFrame containing duplicate matches with columns:
            ["VITS_ID", "ID", "Mean_Cosine_Similarity", "Utterances_Compared"]
    
    Returns:
        List[str]: Unique list of real speaker IDs who had the lowest similarity for each duplicated VITS_ID.
    """
    lowest = (
        duplicates_df
        .sort_values(by="Mean_Cosine_Similarity", ascending=True)
        .groupby("VITS_ID")
        .first()
        .reset_index()
    )
    return lowest["ID"].unique().tolist()


#%%
# === Step 1: Embed Audio ===
real_embeddings  = embed_utterances(REAL_AUDIO_DIR, NUM_UTTERANCES_PER_SPEAKER, "Embedding real")
synth_embeddings = embed_utterances(SYNTH_DIR, desc="Embedding synthetic")

#%%
# === Step 2: Compare Speakers ===
results_each = compute_similarity(real_embeddings, synth_embeddings, use_mean=False)
results_mean = compute_similarity(real_embeddings, synth_embeddings, use_mean=True)

#%%
# === Step 3: Analyse results for per-utterance and mean embeddings ===
best_matches_each, unmatched_each, duplicates_each = analyse_results(results_each)
best_matches_mean, unmatched_mean, duplicates_mean = analyse_results(results_mean)

# === Step 4: Reverse match analysis (for unmatched VITS_IDs) ===
reverse_each = analyse_reverse_matches(results_each, unmatched_each, best_matches_each)
reverse_mean = analyse_reverse_matches(results_mean, unmatched_mean, best_matches_mean)

# === Step 5: Duplicate match analysis (for duplicated VITS_IDs) ===
duplicates_each_df = compare_duplicate_matches(results_each, best_matches_each, duplicates_each)
duplicates_mean_df = compare_duplicate_matches(results_mean, best_matches_mean, duplicates_mean)

# === Step 6: Get lowest-scoring real speaker for each duplicate VITS_ID ===
lowest_ids_each = get_lowest_score_ids(duplicates_each_df)
lowest_ids_mean = get_lowest_score_ids(duplicates_mean_df)

# === Step 7: Print manual review summary for each matching strategy ===
print("\n========== PER-UTTERANCE REPORT ==========")
print("\n🔍 Reverse Matches for Unmatched VITS_IDs (Per-Utterance):")
print(reverse_each.sort_values(by="Reverse_Similarity", ascending=False).to_string(index=False))

print("\n🌀 Duplicate VITS_ID Matches:")
print(duplicates_each_df.sort_values(by=["VITS_ID", "Mean_Cosine_Similarity"], ascending=[True, False]).to_string(index=False))

print("🔻 (Lowest-scoring Real Speakers per Duplicate VITS_ID):")
print(lowest_ids_each)

print("\n========== MEAN-BASED REPORT ==========")
print("\n🔍 Reverse Matches for Unmatched VITS_IDs (Mean-Based):")
print(reverse_mean.sort_values(by="Reverse_Similarity", ascending=False).to_string(index=False))

print("\n🌀 Duplicate VITS_ID Matches:")
print(duplicates_mean_df.sort_values(by=["VITS_ID", "Mean_Cosine_Similarity"], ascending=[True, False]).to_string(index=False))

print("🔻 (Lowest-scoring Real Speakers per Duplicate VITS_ID):")
print(lowest_ids_mean)

# === Step 8: Determine which best matches dataframe to save to disk ===
print("\n\n========== FINAL BEST MATCH SELECTION ==========")

# Compare best_matches from each method
pairs_each = best_matches_each[["ID", "VITS_ID"]].sort_values(by="ID").reset_index(drop=True)
pairs_mean = best_matches_mean[["ID", "VITS_ID"]].sort_values(by="ID").reset_index(drop=True)

if pairs_each.equals(pairs_mean):
    print("✅ Best match results are consistent across both methods.")
    chosen_matches = best_matches_each.copy()
    lowest_ids_to_mask = lowest_ids_each
else:
    print("⚠️ Warning: Best match results differ between per-utterance and mean-based methods!")
    
    # Show mismatched entries
    mismatches = pairs_each[pairs_each["VITS_ID"] != pairs_mean["VITS_ID"]]
    print(f"🔍 Mismatched entries:\n{mismatches}")
    
    # Choose the one with more unique VITS_IDs
    unique_each = best_matches_each["VITS_ID"].nunique()
    unique_mean = best_matches_mean["VITS_ID"].nunique()
    
    if unique_each >= unique_mean:
        print(f"📌 Using per-utterance best matches (unique VITS_IDs: {unique_each})")
        chosen_matches = best_matches_each.copy()
        lowest_ids_to_mask = lowest_ids_each
    else:
        print(f"📌 Using mean-based best matches (unique VITS_IDs: {unique_mean})")
        chosen_matches = best_matches_mean.copy()
        lowest_ids_to_mask = lowest_ids_mean

# Replace VITS_ID with NaN for lowest-scoring duplicates
chosen_matches["VITS_ID"] = chosen_matches.apply(
    lambda row: np.nan if row["ID"] in lowest_ids_to_mask else row["VITS_ID"],
    axis=1
)
removed = chosen_matches[chosen_matches["ID"].isin(lowest_ids_to_mask)]
if not removed.empty:
    print("🗑️ Replaced VITS_ID with NaN for lowest-scoring duplicates:")
    print(removed)

#%%
# === Step 9: Save cleaned best_matches DataFrame ===
chosen_matches.to_csv(MATCHES_FILE, index=False)
print(f"💾 Saved cleaned best matches to {MATCHES_FILE}")
