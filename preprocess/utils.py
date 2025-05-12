import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pyloudnorm as pyln


def normalise_text(text: str) -> str:
    """
    Perform text normalisation by converting all characters to lowercase 
    and collapsing consecutive whitespace characters into a single space. 
    Leading and trailing whitespace is implicitly removed.
    
    Args:
        text (str): The input string to be normalized.
    
    Returns:
        str: A normalised string suitable for consistent processing.
    """
    return " ".join(text.lower().split())  # strip() unnecessary due to split()


def normalise_whitespace(text: str) -> str:
    """
    Normalises whitespace by collapsing multiple spaces, tabs, or newlines
    into a single space. Also trims leading/trailing whitespace.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Cleaned text with consistent single spaces.
    """
    return " ".join(text.split())


def standardise_audio(wav,
                      orig_sr,
                      target_sr,
                      target_lufs=-23,
                      top_db=30):
    """
    Standardise audio by:
    1. Applying LUFS loudness normalisation to ensure consistent perceived loudness.
    2. Applying linear peak limiting (to -1.0 dBFS) to prevent clipping and keep amplitudes within a safe range.
    3. Trimming leading and trailing silence based on a dB threshold.
    4. Resampling to a target sample rate if needed.
    
    Args:
        wav (np.ndarray): Input waveform.
        orig_sr (int): Original sampling rate of the audio.
        target_sr (int): Desired output sample rate (e.g., 16000 Hz).
        target_lufs (float): Target integrated loudness in LUFS (default -23.0).
        top_db (float): Threshold (in decibels) below reference to consider as silence for trimming.
    
    Returns:
        np.ndarray: Standardised waveform.
    """
    # LUFS Loudness normalisation (perceptual)
    meter    = pyln.Meter(orig_sr)  # create LUFS meter
    loudness = meter.integrated_loudness(wav)
    wav      = pyln.normalize.loudness(wav, loudness, target_lufs)
    
    # Apply peak normalisation so that max amplitude is -1 dBFS (≈0.891)
    wav = pyln.normalize.peak(wav, -1.0)
    
    # Trim leading/trailing silence
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    
    # Resample to target sample rate if needed
    if orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
        wav = np.clip(wav, -1.0, 1.0)  # clamp in case of small overshoots
    
    return wav


def save_wav_file(path, wav, sr=16000):
    """
    Save a waveform to disk as 16-bit PCM.
    
    Args:
        path (Path): Output file path.
        wav (np.ndarray): Waveform to save.
        sr (int, optional): Sample rate. Defaults to 16000.
    """
    sf.write(str(path), wav, sr, subtype="PCM_16")


def process_audio_file(input_path: Path, output_path: Path, target_sr=16000):
    """
    Process an audio file by:
    1. Loading the audio file from disk.
    2. Applying LUFS loudness normalisation to ensure consistent perceived loudness.
    3. Applying linear peak limiting (to -1.0 dBFS) to prevent clipping and keep amplitudes within a safe range.
    4. Trimming leading and trailing silence based on a dB threshold.
    5. Resampling to a target sample rate if needed.
    6. Saving to disk at the destingation path.
    """
    # Load the audio waveform and its original sample rate (sr=None)
    wav, orig_sr = librosa.load(str(input_path), sr=None, mono=True)
    
    wav = standardise_audio(wav, orig_sr, target_sr)
    save_wav_file(output_path, wav, sr=target_sr)


def process_wav_array(wav: list,
                      orig_sr: int,
                      output_path: Path,
                      target_sr: int = 16000):
    """
    Process an audio waveform by:
    1. Applying LUFS loudness normalisation to ensure consistent perceived loudness.
    2. Applying linear peak limiting (to -1.0 dBFS) to prevent clipping and keep amplitudes within a safe range.
    3. Trimming leading and trailing silence based on a dB threshold.
    4. Resampling to a target sample rate if needed.
    5. Saving to disk.
    
    Args:
        wav (list or np.ndarray): Raw audio waveform.
        orig_sr (int): Original sample rate of the waveform.
        output_path (Path): Destination path for the processed audio file.
        target_sr (int, optional): Target sample rate in Hz (default: 16000 Hz).
    
    Returns:
        np.ndarray: The processed waveform array.
    """
    # Ensure correct dtype
    wav = np.asarray(wav, dtype=np.float32)
    
    wav = standardise_audio(wav, orig_sr, target_sr)
    save_wav_file(output_path, wav, sr=target_sr)
    return wav


def combine_wav_files(wav_paths, out_path, target_sr=16000, silence_duration=0.0):
    """
    Load multiple WAV files, optionally resample them to a target sample rate,
    insert silence between each clip, and concatenate into a single output file.
    
    This is useful for creating a composite audio file from multiple segments,
    while preserving spacing between utterances (e.g., for synthetic dataset creation).
    
    Args:
        wav_paths (list of str or Path): List of paths to input WAV files.
        out_path (str or Path): Path to save the resulting combined audio file.
        target_sr (int, optional): Target sample rate in Hz. If None, the sample rate
                                   of the first file will be used.
        silence_duration (float, optional): Duration of silence to insert between clips (in seconds).
                                            Must be non-negative. Defaults to 0.0 seconds.
    
    Raises:
        ValueError: If silence_duration is negative.
    """
    if silence_duration < 0:
        raise ValueError("silence_duration must be non-negative")
    
    wavs = []
    for i, path in enumerate(wav_paths):
        # Load audio, resampled to target_sr if specified
        audio, sr = librosa.load(path, sr=target_sr)
        
        # Set target_sr from the first file if not provided
        if target_sr is None:
            target_sr = sr
        
        wavs.append(audio)
        
        # Add silence between clips (except after the last one)
        if i < len(wav_paths) - 1 and silence_duration > 0:
            silence = np.zeros(int(target_sr * silence_duration))
            wavs.append(silence)
    
    # Concatenate all audio and silence
    combined = np.concatenate(wavs)
    
    # Write combined audio to file
    sf.write(out_path, combined, target_sr)
