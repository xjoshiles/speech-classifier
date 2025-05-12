import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import BATCH_SIZE, NUM_WORKERS
from pathlib import Path

SPLIT_MODES = {"tts_system", "speaker", "utterance"}
SUPPORTED_FEATURES = {"raw", "mfcc", "spectrogram_2d", "wav2vec"}


def collate_raw_audio(batch):
    """
    Collate function for raw audio input with attention masks and domain labels.
    """
    waveforms, attn_masks, labels, domains = zip(*batch)
    return {
        "input": torch.stack(waveforms),
        "attention_mask": torch.stack(attn_masks),
        "label": torch.tensor(labels),
        "domain": torch.tensor(domains),
    }


def collate_embeddings(batch):
    """
    Collate function for precomputed embeddings with domain labels.
    """
    embeddings, labels, domains = zip(*batch)
    return {
        "input": torch.stack(embeddings),
        "label": torch.tensor(labels),
        "domain": torch.tensor(domains),
    }


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, dataframe):
        self.samples = [
            (Path(row["path"]), 0 if row["label"] == "real" else 1, int(row["domain"]))
            for _, row in dataframe.iterrows()
        ]
        self.labels = torch.tensor([label for _, label, _ in self.samples])
        self.domains = torch.tensor([domain for _, _, domain in self.samples])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        emb = torch.load(path, weights_only=True)
        return emb, label, domain
    
    def compute_class_weights(self):
        counts = torch.bincount(self.labels, minlength=2).float()
        total = self.labels.numel()
        weights = total / (2 * counts)
        return weights
    
    def compute_domain_weights(self):
        """
        Computes inverse-frequency weights for each domain class.
    
        Returns:
            torch.Tensor: A tensor of shape (num_domains,) containing weights for each domain.
                          Weight for class i = total_samples / (num_domains * count_i)
        """
        num_domains = int(self.domains.max().item()) + 1
        counts = torch.bincount(self.domains, minlength=num_domains).float()
        total = self.domains.numel()
        weights = total / (num_domains * counts)
        return weights


class RawAudioDataset(Dataset):
    def __init__(self, dataframe, max_length=160000, sample_rate=16000):
        # Ensure index is sequential to avoid alignment issues during indexing
        dataframe = dataframe.reset_index(drop=True)
        
        self.paths = dataframe["path"].tolist()
        self.labels = torch.tensor([0 if lbl == "real" else 1 for lbl in dataframe["label"]])
        self.domains = torch.tensor(dataframe["domain"].values)
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        domain = self.domains[idx]
        
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        
        # Ensure waveform shape is [T] (remove channel dimension if present)
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        
        # Sanity check (shouldn't trigger if all files are preprocessed)
        if sr != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sr} for {path}")
        
        length = waveform.size(0)
        
        if length > self.max_length:
            # Truncate
            waveform = waveform[:self.max_length]
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            # Pad with zeros
            pad_len = self.max_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            attention_mask = torch.cat([
                torch.ones(length, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
        
        return waveform, attention_mask, label, domain
    
    def compute_class_weights(self):
        counts = torch.bincount(self.labels, minlength=2).float()
        total = self.labels.numel()
        weights = total / (2 * counts)
        return weights
    
    def compute_domain_weights(self):
        """
        Computes inverse-frequency weights for each domain class.
    
        Returns:
            torch.Tensor: A tensor of shape (num_domains,) containing weights for each domain.
                          Weight for class i = total_samples / (num_domains * count_i)
        """
        num_domains = int(self.domains.max().item()) + 1
        counts = torch.bincount(self.domains, minlength=num_domains).float()
        total = self.domains.numel()
        weights = total / (num_domains * counts)
        return weights


def load_dataloaders(split_mode="speaker", feature_type="wav2vec"):
    """
    Loads train, validation, and test DataLoaders from a pre-split manifest file.
    
    The manifest is expected to be located at:
        data/manifests/split_<split_mode>_<feature_type>.tsv
    
    Each manifest row must contain:
    - 'split': Indicates which set the sample belongs to ("train", "val", or "test")
    - 'path': Path to a precomputed feature file (.pt)
    - 'label': Class label ("real" or "synthetic")
    
    Args:
        split_mode (str): The data splitting strategy used to create the manifest.
                          Must be one of {"random", "speaker", "utterance", "tts_system"}.
        feature_type (str): The type of feature used, e.g., "wav2vec", "mfcc", "spectrogram".
                            Must be one of the supported feature types.
    
    Returns:
        tuple: A tuple of three PyTorch DataLoaders:
            - train_loader: For the training set (with shuffling)
            - val_loader: For the validation set (no shuffling)
            - test_loader: For the test set (no shuffling)
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Invalid split_mode: {split_mode}. Choose from {SPLIT_MODES}.")
    if feature_type not in SUPPORTED_FEATURES:
        raise ValueError(f"Invalid feature_type: {feature_type}. Choose from {SUPPORTED_FEATURES}.")
    
    manifest_path = Path(f"data/manifests/split_{split_mode}_{feature_type}.tsv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    df = pd.read_csv(manifest_path, sep="\t")
    
    required_cols = {"split", "path", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Manifest must include columns: {required_cols}")
    
    # === Extract splits ===
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()
    
    # === Map tts_system strings to numeric domain IDs for Train/Val ===
    train_val_domains = pd.concat([train_df["tts_system"], val_df["tts_system"]])
    unique_train_val_domains = sorted(train_val_domains.unique())
    remap = {name: i for i, name in enumerate(unique_train_val_domains)}
    
    # Apply remap to train and val
    train_df["domain"] = train_df["tts_system"].map(remap)
    val_df["domain"]   = val_df["tts_system"].map(remap)
    
    # For test, assign -1 to unseen domains (those not in remap)
    test_df["domain"] = test_df["tts_system"].map(remap).fillna(-1).astype(int)
    
    print(f"[INFO] Loaded split manifest: {manifest_path.name}")
    print(f"  🟢 Train: {len(train_df)}")
    print(f"  🟡 Val  : {len(val_df)}")
    print(f"  🔴 Test : {len(test_df)}")
    
    # Show remapping
    print(f"[INFO] Domain remap for Training/Validation (only used for DAT):")
    for name, new_id in remap.items():
        print(f"  {name:<15} → {new_id}")
    
    # Choose dataset class based on whether we're fine-tuning
    if feature_type == "raw":
        dataset_cls = RawAudioDataset
    else:
        dataset_cls = PrecomputedEmbeddingDataset
    
    # Get the collate_fn for this feature type
    collate_fn = collate_raw_audio if feature_type == "raw" else collate_embeddings
    
    train_loader = DataLoader(
        dataset_cls(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset_cls(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset_cls(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

