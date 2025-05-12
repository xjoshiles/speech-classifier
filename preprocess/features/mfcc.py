import torch
import torchaudio


class MFCCExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=40, device="cuda"):
        # Determine whether to use GPU or CPU for inference
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc  # number of MFCC coefficients to compute
        
        # Torchaudio MFCC transformation pipeline
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
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
        
        # shape: (1, n_mfcc, time)
        mfcc = self.transform(waveform).squeeze(0).transpose(0, 1)  # → shape: (time, n_mfcc)
        
        # Pad/truncate to fixed length so all feature tensors have a uniform shape
        if max_frames is not None:
            if mfcc.size(0) < max_frames:
                pad_len = max_frames - mfcc.size(0)
                mfcc = torch.cat([mfcc, torch.zeros(pad_len, self.n_mfcc).to(self.device)], dim=0)
            else:
                mfcc = mfcc[:max_frames]
        
        return mfcc
    
    def extract_and_save(self, wav_path, emb_path, max_frames=320):
        features = self.extract(wav_path, max_frames=max_frames)
        torch.save(features.cpu(), emb_path)
