from .cnn import CNNClassifier, CNN_DAT, CNN2DClassifier, CNN2D_DAT
from inspect import signature

from .logistic import LogisticRegression
from .lstm import LSTMClassifier, LSTM_DAT
from .mlp import MLPClassifier
from .wav2vec import FineTunedWav2VecClassifier, FineTunedWav2Vec_DAT


MODEL_REGISTRY = {
    "cnn": CNNClassifier,
    "cnn_dat": CNN_DAT,
    "cnn_2d": CNN2DClassifier,
    "cnn_2d_dat": CNN2D_DAT,
    "logistic": LogisticRegression,
    "lstm": LSTMClassifier,
    "lstm_dat": LSTM_DAT,
    "mlp": MLPClassifier,
    "wav2vec": FineTunedWav2VecClassifier,
    "wav2vec_dat": FineTunedWav2Vec_DAT
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY)}")
    
    model_cls = MODEL_REGISTRY[name]
    
    # Remove input_dim if the model doesn't accept it
    if "input_dim" in kwargs:
        sig = signature(model_cls)
        if "input_dim" not in sig.parameters:
            kwargs.pop("input_dim")
    
    return model_cls(**kwargs)
