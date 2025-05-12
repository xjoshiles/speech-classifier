from .mfcc import MFCCExtractor
from .spectrogram import Spectrogram2DExtractor
from .wav2vec import Wav2VecExtractor

EXTRACTOR_MAP = {
    "mfcc": MFCCExtractor,
    "spectrogram_2d": Spectrogram2DExtractor,
    "wav2vec": Wav2VecExtractor,
}


def get_extractor(name: str, device: str = "cpu", **kwargs):
    """
    Retrieve an extractor class instance by name.

    Args:
        name (str): Feature type ('mfcc', 'spectrogram1d', 'spectrogram2d', 'wav2vec').
        device (str): Device to use ('cpu' or 'cuda').
        **kwargs: Additional parameters passed to the extractor.

    Returns:
        An instantiated extractor object.

    Raises:
        ValueError: If the feature type is not supported.
    """
    name = name.lower()
    if name not in EXTRACTOR_MAP:
        raise ValueError(f"Unsupported feature type: {name}. Available: {list(EXTRACTOR_MAP.keys())}")
    
    return EXTRACTOR_MAP[name](device=device, **kwargs)
