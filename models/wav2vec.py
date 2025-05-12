import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from .utils.gradient import GradientReversalLayer


class FineTunedWav2VecClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name="facebook/wav2vec2-base",
                 hidden_dim=128,
                 dropout=0.3,
                 freeze_feature_extractor=True):
        super().__init__()
        
        self.requires_attention_mask = True
        
        # Load pretrained Wav2Vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Optionally freeze the feature extractor (the CNN frontend)
        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()
        
        encoder_dim = self.encoder.config.hidden_size  # Typically 768
        
        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, input_values, attention_mask=None):
        # input_values: [B, T] (raw waveforms)
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T', D]
        
        # Compute attention scores and weights
        attn_scores = self.attention(hidden_states)               # [B, T', 1]
        attn_weights = torch.softmax(attn_scores, dim=1)          # [B, T', 1]
        
        # Weighted sum (attention pooling)
        pooled = torch.sum(attn_weights * hidden_states, dim=1)   # [B, D]
        
        return self.classifier(pooled)


class FineTunedWav2Vec_DAT(nn.Module):
    def __init__(self,
                 pretrained_model_name="facebook/wav2vec2-base",
                 hidden_dim=128,
                 dropout=0.3,
                 freeze_feature_extractor=True,
                 num_domains=5,
                 lambda_grl=1.0):
        super().__init__()
        
        self.requires_attention_mask = True
        
        # Load Wav2Vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Optionally freeze the feature extractor (the CNN frontend)
        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()
        
        self.encoder_dim = self.encoder.config.hidden_size  # e.g. 768
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        
        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(self.encoder_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Domain discriminator head
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        
        attn_scores = self.attention(hidden_states)              # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)         # [B, T, 1]
        pooled = torch.sum(attn_weights * hidden_states, dim=1)  # [B, D]
        
        # Main output
        class_output = self.classifier(pooled)
        
        # Domain output via GRL
        domain_output = self.domain_discriminator(self.grl(pooled))
        
        return class_output, domain_output
