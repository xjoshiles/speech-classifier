# 🗣️ Real vs. Synthetic Speech Classifier

This Streamlit application allows users to upload speech audio files and classify them as either human or AI-generated using pre-trained machine learning models. It supports `.wav`, `.flac`, `.m4a`, `.mp3`, and `.ogg` formats and includes visualisations such as waveform, MFCCs, and spectrograms.

---

## 🔧 Installation & Setup

You can run this application using **Conda**, **pip**, or **Docker**.

---

### ✅ Option 1: Using Conda

1. Create and activate a new environment:
    ```bash
    conda create -n speech-classifier python=3.9 -y
    conda activate speech-classifier
    ```

2. Install `pip` and dependencies:
    ```bash
    conda install pip
    pip install -r requirements.txt
    ```

3. (Optional) Ensure `ffmpeg` is installed for `.m4a` file support:
    - With conda:
      ```bash
      conda install -c conda-forge ffmpeg
      ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

---

### ✅ Option 2: Using pip (Virtualenv or System Python)

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

### ✅ Option 3: Using Docker (for isolated reproducibility)

1. Build the Docker image:
    ```bash
    docker build -t speech-classifier .
    ```

2. Run the container:
    ```bash
    docker run -p 8501:8501 speech-classifier
    ```

3. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧭 How to Use the Interface

### 1. **Select a Model**
Use the dropdown menu on the left sidebar to choose one of the pretrained models

### 2. **Upload an Audio File**
Drag and drop or browse to upload a file.

- **Supported formats**: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`
- **Maximum size**: 200MB

### 3. **View the Results**
After upload, the app will:

- Automatically convert, preprocess, and standardise the audio
- Extract the corresponding features
- Run the selected model on the extracted features
- Display a classification: **Real** or **Synthetic**
- Show a confidence bar with predicted probability

### 4. **Visualise Results**
The following plots are generated:

- **Waveform**: amplitude over time
- **Spectrogram**: time-frequency representation
- **MFCC**: Mel-frequency cepstral coefficients

---

## 📂 File Structure

### 🧭 App Components

Files relevant to running the Streamlit classifier (main app).

```
.
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── data/                 # Some sample files to try
├── models/               # Model classes
├── results/              # Trained models and configs
├── preprocess/           # Audio standardisation and feature extraction code
└── README.md             # This file
```


### 🛠️ Project Pipeline (for research reproducibility)

Scripts and utilities for end-to-end system development: from data preparation to model training and evaluation. Detailed explanations can be found by inspecting each script, though a high-level overview of their purpose has also been included here.

```
.
├── data/           # Raw + processed speech data
│   ├── corpora/    # Original directory for the VCTK corpus
│   ├── processed/  # Resampled and normalised audio
│   └── manifests/  # Metadata for train, val, test splits for all features
│
├── config.py       # Shared constants and configuration settings
├── dataset.py      # Dataset loading and preparation for PyTorch training
├── evaluate.py     # Model evaluation: loss, accuracy, etc.
├── grid_search.py  # Hyperparameter tuning across model-feature combinations
├── train.py        # Train the given model (supports attention mask, DAT)
│
├── preprocess/                      # Data processing and synthesis scripts
│   ├── extract_features.py          # Extract MFCC, spectrogram, wav2vec embeddings
│   ├── generate_split_manifests.py  # Create speaker/system-disjoint train/val/test splits
│   ├── generate_unified_manifests.py
│   ├── synthesize_fastpitch.py
│   ├── synthesize_openvoice.py
│   ├── synthesize_vits.py
│   ├── synthesize_xtts.py
│   ├── synthesize_yourtts.py
│   ├── utils.py                              # Helper functions
│   ├── vctk_fix_transcript_case.py           # Fixing VCTK corpus metadata
│   ├── vctk_fix_transcript_whitespace.py     # Fixing VCTK corpus metadata
│   ├── vctk_generate_transcript_ids.py       # For matching samples on spoken content
│   ├── vctk_match_vits_speakers.py           # Match VCTK/VITS speakers due to bug
│   ├── vctk_resample.py                      # Standardise + rename VCTK corpus
│   ├── vctk_synthesize_vits_for_matching.py
│   └── vctk_update_speaker_info.py           # Fixing VCTK corpus metadata
│
├── visualise/                      # Plotting and visualisation
│   ├── lufs_visualisation.py       # Comparing LUFS loudness across domains
│   ├── plots.py                    # Loss curves, ROC curves, Confusion Matrices

```

---

## 📦 Data and Models

Due to the large size of the full dataset (approximately **50GB**, including all extracted features), only a subset is included in this repository:

- ✅ **Raw Audio Subset**:  
  For reproducibility and testing, **10 speakers** with **5 utterances** per speaker have been included across both real and synthetic domains. These are stored in the `data/processed` directory.

- ✅ **Model Subset**:  
  Rather than include all trained models (~several hundred across feature and architecture combinations), only the **top-performing model** for each feature/model pairing has been retained in the `results/` folder. Note however that since the Fine-Tuned Wav2Vec model was `~360MB` and exhibited poor performance due to poor training, it was also dropped from the model subset.

---

## 📦 Requirements

- Python 3.9
- Streamlit ≥ 1.36
- PyTorch, torchaudio
- Hugging Face transformers
- Librosa, Scipy, Scikit-learn
- FFmpeg (for `.m4a` support via `pydub`)

---

## 👤 Author

This application was developed as part of a final-year dissertation on **AI-generated speech detection**.

For questions or feedback, please contact ji91@canterbury.ac.uk.

---
## 🔗 Project Repository

Source code: [https://github.com/xjoshiles/speech-classifier](https://github.com/xjoshiles/speech-classifier)
