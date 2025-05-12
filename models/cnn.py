import torch.nn as nn
from .utils.gradient import GradientReversalLayer


class CNNClassifier(nn.Module):
    """
    1D Convolutional classifier for time-series or sequence-based features (e.g., MFCCs, Wav2Vec embeddings).
    Outputs binary logits for real vs. synthetic speech detection.
    """
    def __init__(self,
                 input_dim=768,
                 num_filters=128,
                 kernel_size=5,
                 num_conv_layers=1,
                 dropout=0.0,
                 pooling="adaptive",
                 activation_fn="relu",
                 hidden_dim=128):
        super().__init__()
        
        self.activation_fn = self._get_activation(activation_fn)
        self.pooling_type = pooling
        self.dropout = nn.Dropout(dropout)
        
        # === Convolutional feature extractor ===
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = input_dim if i == 0 else num_filters
            conv_layers.append(nn.Conv1d(in_ch, num_filters, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(self.activation_fn)
        self.conv_block = nn.Sequential(*conv_layers)
        
        # === Pooling strategy ===
        if pooling == "adaptive":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "max":
            self.pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), nn.AdaptiveMaxPool1d(1))
        elif pooling == "avg":
            self.pool = nn.Sequential(nn.AvgPool1d(kernel_size=2), nn.AdaptiveMaxPool1d(1))
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, 2)
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
        x = x.transpose(1, 2)  # to (batch, channels, seq_len)
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)  # shape: (batch, channels)
        return self.classifier(x)


class CNN_DAT(nn.Module):
    """
    1D CNN with Domain-Adversarial Training (DAT) for speaker- or system-invariant classification.
    Outputs:
        - class_output: binary logits (real vs synthetic)
        - domain_output: logits for domain classification (e.g., TTS system)
    """
    def __init__(self,
                 input_dim=768,
                 num_filters=128,
                 kernel_size=5,
                 num_conv_layers=1,
                 dropout=0.0,
                 pooling="adaptive",
                 activation_fn="relu",
                 hidden_dim=128,
                 num_domains=5,
                 lambda_grl=1.0):
        super().__init__()
        
        self.activation_fn = self._get_activation(activation_fn)
        self.pooling_type = pooling
        self.dropout = nn.Dropout(dropout)
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        
        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = input_dim if i == 0 else num_filters
            conv_layers.append(nn.Conv1d(in_ch, num_filters, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(self.activation_fn)
        self.conv_block = nn.Sequential(*conv_layers)
        
        if pooling == "adaptive":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "max":
            self.pool = nn.Sequential(nn.MaxPool1d(kernel_size=2), nn.AdaptiveMaxPool1d(1))
        elif pooling == "avg":
            self.pool = nn.Sequential(nn.AvgPool1d(kernel_size=2), nn.AdaptiveMaxPool1d(1))
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, 2)
        )
        
        self.domain_discriminator = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, num_domains)
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
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x), self.domain_discriminator(self.grl(x))


class CNN2DClassifier(nn.Module):
    """
    2D CNN classifier for time-frequency inputs (e.g., spectrograms).
    """
    def __init__(self,
                 input_dim=768,
                 num_filters=64,
                 kernel_size=5,
                 num_conv_layers=2,
                 dropout=0.3,
                 pooling="adaptive",
                 activation_fn="relu",
                 hidden_dim=128):
        super().__init__()
        
        self.activation_fn = self._get_activation(activation_fn)
        self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling
        
        layers = []
        for i in range(num_conv_layers):
            in_ch = input_dim if i == 0 else num_filters
            layers.append(nn.Conv2d(in_ch, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(self.activation_fn)
        
        self.conv_block = nn.Sequential(*layers)
        
        if pooling == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, 2)
        )
    
    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x_flat = x.view(x.size(0), -1)
        return self.classifier(x_flat)


class CNN2D_DAT(nn.Module):
    """
    2D CNN with Domain-Adversarial Training (DAT) for time-frequency inputs.
    """
    def __init__(self,
                 input_dim=768,
                 num_filters=64,
                 kernel_size=5,
                 num_conv_layers=2,
                 dropout=0.3,
                 pooling="adaptive",
                 activation_fn="relu",
                 hidden_dim=128,
                 num_domains=5,
                 lambda_grl=1.0):
        super().__init__()
        
        self.activation_fn = self._get_activation(activation_fn)
        self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        
        layers = []
        for i in range(num_conv_layers):
            in_ch = input_dim if i == 0 else num_filters
            layers.append(nn.Conv2d(in_ch, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(self.activation_fn)
        
        self.conv_block = nn.Sequential(*layers)
        
        if pooling == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, 2)
        )
        
        self.domain_discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            self.activation_fn,
            self.dropout,
            nn.Linear(hidden_dim, num_domains)
        )
    
    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        x_flat = x.view(x.size(0), -1)
        class_output = self.classifier(x_flat)
        domain_output = self.domain_discriminator(self.grl(x_flat))
        return class_output, domain_output
