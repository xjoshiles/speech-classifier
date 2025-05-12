# Paths
from pathlib import Path
HERE = Path(__file__).parent

# Data
BATCH_SIZE = 16
NUM_WORKERS = 2

# Training
EPOCHS = 15
LEARNING_RATE = 1e-4
USE_GPU = True  # Set to False to force CPU
