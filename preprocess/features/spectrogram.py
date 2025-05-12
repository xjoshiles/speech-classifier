import torch
import torchaudio


class Spectrogram2DExtractor:
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, power=2.0, device="cuda"):
        # Choose device based on CUDA availability
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power  # 1.0 for magnitude, 2.0 for power spectrogram
        
        # Torchaudio Spectrogram transformation + log scale pipeline
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.power,
                normalized=False
            ),
            torchaudio.transforms.AmplitudeToDB()  # converts amplitude/power to log scale
        ).to(self.device)
    
    @torch.no_grad()
    def extract(self, wav_path, max_frames=320):
        # Load the audio waveform using torchaudio
        waveform, sr = torchaudio.load(wav_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if sample rate mismatch       
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        
        # Move waveform tensor to device
        waveform = waveform.to(self.device)
        
        # Output shape: [1, freq_bins, time]
        spec = self.transform(waveform).squeeze(0)  # shape: [freq, time]
        
        # Pad/truncate to fixed length so all feature tensors have a uniform shape
        if max_frames is not None:
            if spec.size(1) < max_frames:
                pad_len = max_frames - spec.size(1)
                spec = torch.cat([spec, torch.zeros(spec.size(0), pad_len).to(self.device)], dim=1)
            else:
                spec = spec[:, :max_frames]
        
        # Final shape for CNN2D: [1, freq_bins, time]
        spec = spec.unsqueeze(0)

        return spec
    
    def extract_and_save(self, wav_path, emb_path, max_frames=320):
        features = self.extract(wav_path, max_frames=max_frames)
        torch.save(features.cpu(), emb_path)

