import torch
import torchaudio
from torch.amp import autocast
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import logging as hf_logging


class Wav2VecExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base", device="cuda"):
        # Silence Hugging Face weight initialisation warnings as we only use
        # the encoder output (last_hidden_state), not the uninitialised parts
        hf_logging.set_verbosity_error()
        
        # Determine whether to use GPU or CPU for inference
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model + processor from HuggingFace
        # to(device) puts the model on the GPU or CPU
        # eval() puts the model in inference mode (no dropout, etc.)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()
        
        self.hidden_size = self.model.config.hidden_size  # i.e. 768
    
    @torch.no_grad()
    def extract(self, wav_path, max_frames=320):
        # Load the audio waveform using torchaudio
        waveform, sr = torchaudio.load(wav_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if sample rate mismatch       
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            sr = 16000
        
        # Preprocess waveform into model-ready input tensors and move to device
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=sr,
            return_tensors="pt").input_values.to(self.device)
        
        # Extract the last hidden layer: shape (T, 768), where T is the number
        # of time frames (depends on audio length and Wav2Vec2's downsampling)
        if self.device.type == "cuda":
            
            # Automatic mixed-precision (AMP) mode on GPU for speed (Tensor cores)
            with autocast(device_type="cuda"):
                features = self.model(inputs).last_hidden_state.squeeze(0)
        else:
            features = self.model(inputs).last_hidden_state.squeeze(0)
        
        # Pad/truncate to fixed length so all feature tensors have a uniform shape
        if features.size(0) < max_frames:
            pad_len = max_frames - features.size(0)
            features = torch.cat([features, torch.zeros(pad_len, self.hidden_size).to(features.device)], dim=0)
        else:
            features = features[:max_frames]
        
        return features
    
    def extract_and_save(self, wav_path, emb_path, max_frames=320):
        features = self.extract(wav_path, max_frames=max_frames)
        torch.save(features.cpu(), emb_path)
