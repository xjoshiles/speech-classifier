import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tempfile
from pydub import AudioSegment
from pathlib import Path

from models import get_model
from preprocess.utils import process_wav_array
from preprocess.features import MFCCExtractor, Spectrogram2DExtractor, Wav2VecExtractor

# === Configuration ===
HERE   = Path(__file__).parent
MODELS = pd.read_csv(HERE / 'results/best_models.csv')

FEATURE_INPUT_DIMS = {
    "mfcc": 40,
    "wav2vec": 768,
    "spectrogram_2d": 1,
    "raw": 160000
}

# === Page Title ===
st.set_page_config(page_title="Real vs. Synthetic Speech Classifier", layout="centered")
st.markdown(
    "<h1 style='font-size: 36px;'>🗣️ Real vs Synthetic Speech Classifier</h1>",
    unsafe_allow_html=True
)

# First dropdown: select feature type
features = sorted(MODELS["Features"].unique())
selected_feature = st.sidebar.selectbox("Select feature type:", features)

# Filter models by selected feature
models_for_feature = MODELS[MODELS["Features"] == selected_feature]

# Second dropdown: select model
model_options = {
    row['Model']: HERE / row['Path']
    for _, row in models_for_feature.iterrows()
}
selected_model = st.sidebar.selectbox("Select model:", model_options.keys())
model_dir = model_options[selected_model]
feature = model_dir.parent.name


# === Load model ===
@st.cache_resource
def load_model(model_dir, feature_type, device="cpu"):
    """
    Load a trained PyTorch model from a specified directory.
    
    This function loads the model configuration, constructs the appropriate model 
    architecture using `get_model()`, loads the trained weights, and sets the model 
    to evaluation mode.
    
    Args:
        model_dir (str or Path): Path to the directory containing `config.json` and `model.pt`.
        feature_type (str): Type of input features used by the model (e.g., "mfcc", "wav2vec").
        device (str): Device to load the model onto ("cpu" or "cuda").
    
    Returns:
        torch.nn.Module: The loaded and ready-to-infer PyTorch model.
    """
    model_dir = Path(model_dir)
    
    # Load model config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    model_name = config.pop("model")  # remove 'model' key for get_model()
    
    # Handle input_dim argument for models (not saved in config)
    if "input_dim" not in config and feature_type in FEATURE_INPUT_DIMS:
        input_dim = FEATURE_INPUT_DIMS[feature_type]
        if input_dim is not None:
            config["input_dim"] = input_dim
    
    # Load the model architecture
    model = get_model(model_name, **config).to(device)

    # Load the model weights
    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode (disables dropout)
    
    return model


model = load_model(model_dir, feature)

# === File Upload ===
st.markdown(f"**Loaded Model:** `{selected_model} ({selected_feature})`")
audio_file = st.file_uploader("Upload a WAV file", type=["wav", "flac", "m4a", "mp3", "ogg"])
st.audio(audio_file)

if audio_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        suffix = Path(audio_file.name).suffix
        
        # === Save uploaded file to temporary path ===
        temp_path_orig = tmpdir / f"uploaded{suffix}"
        with open(temp_path_orig, "wb") as f:
            f.write(audio_file.read())
        
        # === If it's an M4A, convert it to WAV ===
        if temp_path_orig.suffix.lower() == ".m4a":
            converted_path = tmpdir / "converted.wav"
            audio = AudioSegment.from_file(temp_path_orig)
            audio.export(converted_path, format="wav")
            temp_path_orig = converted_path
        
        # === Load the audio waveform using torchaudio ===
        waveform_orig, sr = torchaudio.load(temp_path_orig)
        
        # === Convert to mono if stereo ===
        if waveform_orig.size(0) > 1:
            waveform_orig = waveform_orig.mean(dim=0)
        
        # === Convert to NumPy array ===
        waveform_orig = waveform_orig.squeeze()
        
        # === Standardise waveform ===
        # (LUFS, Linear Peak Limiting, Silence Trimming, Resampling)
        temp_path_stnd = tmpdir / "standardised.wav"
        waveform = process_wav_array(
            waveform_orig, orig_sr=sr, output_path=temp_path_stnd, target_sr=16000)
        
        # === Extract MFCC & Spectrogram from Original Audio for Display ===
        mfcc_extractor = MFCCExtractor(sr, device='cpu')
        spec_extractor = Spectrogram2DExtractor(sr, device='cpu')
        mfcc = mfcc_extractor.extract(temp_path_orig, max_frames=None)
        spec = spec_extractor.extract(temp_path_orig, max_frames=None)
        
        # === Extract Features from Standardised Audio for Model Prediction ===
        if feature == "mfcc":
            model_mfcc_extractor = MFCCExtractor(sample_rate=16000, device='cpu')
            feature_input = model_mfcc_extractor.extract(temp_path_stnd)
        elif feature == "wav2vec":
            wav2vec_extractor = Wav2VecExtractor(device='cpu')
            feature_input = wav2vec_extractor.extract(temp_path_stnd)
        elif feature == "spectrogram_2d":
            model_spec_extractor = Spectrogram2DExtractor(sample_rate=16000, device='cpu')
            feature_input = model_spec_extractor.extract(temp_path_stnd)
        else:
            feature_input = torch.from_numpy(waveform).float()
            
            # Ensure 1D shape: [T]
            if feature_input.ndim > 1:
                feature_input = feature_input.squeeze()
            
            length = feature_input.size(0)
            
            if length > 160000:
                # Truncate
                feature_input = feature_input[:160000]
                attn_mask = torch.ones(160000, dtype=torch.long)
            else:
                # Pad with zeros
                pad_len = 160000 - length
                feature_input = torch.nn.functional.pad(feature_input, (0, pad_len))
                attn_mask = torch.cat([
                    torch.ones(length, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long)
                ])
        
        # Add batch dimension
        feature_input = feature_input.unsqueeze(0)
        
        # === Run Prediction ===
        with torch.no_grad():
            result = (
                model(feature_input, attention_mask=attn_mask.unsqueeze(0))
                if hasattr(model, "requires_attention_mask") and model.requires_attention_mask
                else model(feature_input)
            )
            if isinstance(result, tuple):
                result = result[0]
            probs = torch.softmax(result, dim=1).squeeze().numpy()
        
        label = "Real" if np.argmax(probs) == 0 else "Synthetic"
        st.markdown(f"<h3>Prediction: {label}</h3>", unsafe_allow_html=True)
        st.progress(float(probs[1]))
        st.write(f"Confidence: {probs[1]*100:.2f}% for synthetic")
        
        # === Display Waveform ===
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(10, 4))
        time_axis = np.linspace(0, len(waveform_orig) / sr, num=len(waveform_orig))
        ax.plot(time_axis, waveform_orig)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.margins(x=0)  # remove horizontal padding to align with other plots
        st.pyplot(fig)
        
        # === Display Spectrogram ===
        st.subheader("Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        spec_np = spec.squeeze(0).numpy()
        num_frames = spec_np.shape[1]
        hop_length = 160
        frame_duration = hop_length / sr
        extent = [0, num_frames * frame_duration, 0, sr / 2]
        
        ax.imshow(spec_np, origin="lower", aspect="auto", cmap="viridis", extent=extent)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        st.pyplot(fig)
        
        # === Display MFCC ===
        st.subheader("MFCC")
        fig, ax = plt.subplots(figsize=(10, 4))
        mfcc_np = mfcc.T.numpy()
        num_frames = mfcc_np.shape[1]
        hop_length = 160
        frame_duration = hop_length / sr
        extent = [0, num_frames * frame_duration, 0, mfcc_np.shape[0]]
        
        ax.imshow(mfcc_np, origin="lower", aspect="auto", cmap="viridis", extent=extent)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC Coefficients")
        st.pyplot(fig)
