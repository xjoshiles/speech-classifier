import torch
import torch.nn as nn
from .utils.gradient import GradientReversalLayer


class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=128,
                 num_layers=1,
                 dropout=0.2,
                 bidirectional=True,
                 num_classes=2):
        super().__init__()
        
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1
        
        # LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only applies if num_layers > 1
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):  # x: [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers * num_directions, B, hidden]
        
        if self.bidirectional:
            # Concatenate final forward and backward hidden states from last layer
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2*hidden_dim]
        else:
            h_final = h_n[-1]  # [B, hidden_dim]

        return self.classifier(h_final)  # [B, num_classes]


class LSTM_DAT(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=128,
                 num_layers=1,
                 dropout=0.2,
                 bidirectional=True,
                 num_classes=2,
                 num_domains=5,
                 lambda_grl=1.0):
        super().__init__()
        
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        
        # LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only applies if num_layers > 1
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Domain discriminator head
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains)
        )
    
    def forward(self, x):  # x: [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers * num_directions, B, hidden]
        
        if self.bidirectional:
            # Concatenate final forward and backward hidden states from last layer
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2*hidden_dim]
        else:
            h_final = h_n[-1]  # [B, hidden_dim]
        
        class_output = self.classifier(h_final)  # [B, num_classes]
        domain_output = self.domain_discriminator(self.grl(h_final))
        
        return class_output, domain_output

