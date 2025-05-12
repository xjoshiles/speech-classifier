import torch
import torch.nn as nn
from .utils.gradient import GradientReversalLayer


class MLPClassifier(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dims=[128],
                 dropout=0.2,
                 activation_fn="relu",
                 num_classes=2):
        super().__init__()
        
        activ = self._get_activation(activation_fn)
        
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activ)
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
    
    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x):
        # Average time dimension: [B, T, D] → [B, D]
        x = x.mean(dim=1)
        return self.net(x)


class MLP_DAT(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dims=[128],
                 dropout=0.2,
                 activation_fn="relu",
                 num_classes=2,
                 num_domains=5,
                 lambda_grl=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        self.hidden_size = hidden_dims[-1]
        
        activ = self._get_activation(activation_fn)
        
        # Shared feature extractor
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activ)
            layers.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*layers)
        
        # Task classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Domain discriminator head
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains)
        )
    
    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x):
        # Average time dimension: [B, T, D] → [B, D]
        x = x.mean(dim=1)  # Temporal average pooling
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        domain_output = self.domain_discriminator(self.grl(features))
        return class_output, domain_output
